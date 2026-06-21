"""
Dashboard interativo de fenologia.

Uso (a partir de fenologia/):
    source .venv/bin/activate
    uvicorn dashboard.app:app --port 8000 --reload

Acesse http://localhost:8000
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import h3
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent          # fenologia/
PHENO_PATH = ROOT / "data/output/fenologia_brasil_knn.parquet"
PARTS_DIR  = ROOT / "data/output/_parts"
HTML_PATH  = Path(__file__).parent / "index.html"

# Adiciona fenologia/ ao path para importar fenologia.phenophase
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_CYCLE_DAYS = {
    "soja": 120, "cana": 150, "arroz": 100, "algodao": 110,
    "cafe": 120, "citrus": 120, "dende": 120,
    "outras_lavouras_temporarias": 100, "outras_lavouras_perenes": 120,
    "segunda_safra": 90, "segunda_safra_algodao": 100,
    "segunda_safra_outras_temporarias": 90,
}

MAX_CYCLE_DAYS = {
    "soja": 160, "cana": 420, "arroz": 160, "algodao": 220,
    "cafe": 400, "citrus": 400, "dende": 400,
    "outras_lavouras_temporarias": 200, "outras_lavouras_perenes": 400,
    "segunda_safra": 150, "segunda_safra_algodao": 220,
    "segunda_safra_outras_temporarias": 180,
}

app = FastAPI(title="Fenologia Brasil")

_pheno: Optional[pd.DataFrame] = None
_ts: Optional[pd.DataFrame] = None
_grid_cache: dict = {}
_cycles_cache: dict = {}   # (hex_id, cultura) → cycles response


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _pheno, _ts
    log.info("Carregando fenologia_brasil_knn.parquet …")
    _pheno = pd.read_parquet(PHENO_PATH)
    log.info(f"  {len(_pheno):,} registros fenológicos")

    log.info("Carregando séries temporais (_parts/) …")
    parts = sorted(PARTS_DIR.glob("*.parquet"))
    dfs = []
    for p in parts:
        try:
            df = pd.read_parquet(p)
            if len(df) > 0 and "id_hexagono" in df.columns:
                dfs.append(df)
        except Exception:
            continue
    _ts = pd.concat(dfs, ignore_index=True)
    _ts = _ts.drop_duplicates(subset=["id_hexagono", "data"])
    _ts["data"] = pd.to_datetime(_ts["data"])
    _ts = _ts.set_index("id_hexagono", drop=False)
    _ts.index.name = "_idx"
    log.info(f"  {len(_ts):,} observações, {_ts['id_hexagono'].nunique():,} hexágonos únicos")
    log.info("Pronto → http://localhost:8000")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PATH.read_text("utf-8"))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/culturas")
def api_culturas():
    out: dict[str, list[str]] = {}
    for (c, t), _ in _pheno.groupby(["cultura", "tipo_safra"]):
        out.setdefault(c, []).append(t)
    return out


@app.get("/api/grid")
def api_grid(cultura: str, tipo: str):
    key = (cultura, tipo)
    if key not in _grid_cache:
        sub = _pheno[(_pheno["cultura"] == cultura) & (_pheno["tipo_safra"] == tipo)]
        features = []
        for _, row in sub.iterrows():
            try:
                bnd    = h3.cell_to_boundary(row["id_hexagono"])
                coords = [[lon, lat] for lat, lon in bnd]
                coords.append(coords[0])
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "id":       row["id_hexagono"],
                        "sos_doy":  round(float(row["sos_doy"]),  1),
                        "pos_doy":  round(float(row["pos_doy"]),  1),
                        "eos_doy":  round(float(row["eos_doy"]),  1),
                        "n_ciclos": int(row["n_ciclos"]),
                        "r2":       round(float(row["r2_medio"]), 3),
                        "knn":      bool(row.get("interpolated", False)),
                    },
                })
            except Exception:
                continue
        _grid_cache[key] = {"type": "FeatureCollection", "features": features}
        log.info(f"grid {cultura}/{tipo}: {len(features):,} hexágonos")

    return JSONResponse(content=_grid_cache[key])


@app.get("/api/cycles")
def api_cycles(hex_id: str, cultura: str):
    """
    Re-roda extract_phenometrics para o hexágono e retorna:
      - série EVI bruta + suavizada
      - por ciclo: gaussiana, SOS/POS/EOS efetivos, R²
    """
    cache_key = (hex_id, cultura)
    if cache_key in _cycles_cache:
        return JSONResponse(_cycles_cache[cache_key])

    from fenologia.phenophase import extract_phenometrics, adaptive_smoothing

    col = f"evi_medio_{cultura}"
    if hex_id not in _ts.index:
        raise HTTPException(404, "Hexágono não encontrado na série temporal")

    df = (_ts.loc[[hex_id], ["data", col]]
            .dropna(subset=[col])
            .sort_values("data"))

    if len(df) < 10:
        raise HTTPException(422, "Série temporal muito curta para detecção de ciclos")

    min_days = MIN_CYCLE_DAYS.get(cultura, 90)
    max_days = MAX_CYCLE_DAYS.get(cultura, 400)

    # DataFrame no formato esperado pelo phenophase
    ts_df = df.rename(columns={"data": "datetime", col: "NDVI_mean"}).copy()

    result = extract_phenometrics(
        ts_df,
        ndvi_column="NDVI_mean",
        min_cycle_length_days=min_days,
        smoothing_method="both",
        quality_threshold=0.85,
    )

    # Série suavizada (mesmo método do phenophase)
    ndvi_arr  = ts_df["NDVI_mean"].values.astype(float)
    dates_arr = ts_df["datetime"].values
    ndvi_smooth = adaptive_smoothing(ndvi_arr, dates_arr, method="both")

    dates_str = [str(d)[:10] for d in df["data"]]

    # Serializa ciclos bem-sucedidos dentro dos limites de duração
    cycles_out = []
    for c in result.get("cycles", []):
        if not c.get("fit_success"):
            continue
        if not (min_days <= c["cycle_length_days"] <= max_days):
            continue
        gp = c["gaussian_params"]
        ph = c["phenophase_dates"]
        pv = c["phenophase_values"]

        # Gera pontos densos da gaussiana (a cada 4 dias no intervalo do ciclo)
        cs = pd.Timestamp(c["cycle_start"])
        ce = pd.Timestamp(c["cycle_end"])
        n_pts = max(60, int(c["cycle_length_days"] / 3))
        gauss_dates, gauss_vals = [], []
        for i in range(n_pts + 1):
            t = cs + pd.Timedelta(days=i * c["cycle_length_days"] / n_pts)
            if t > ce:
                break
            x = (t - cs).total_seconds() / 86400
            y = (gp["amplitude"]
                 * np.exp(-0.5 * ((x - gp["mean_days"]) / gp["std_dev_days"]) ** 2)
                 + gp["offset"])
            gauss_dates.append(str(t)[:10])
            gauss_vals.append(round(float(y), 4))

        cycles_out.append({
            "cycle_num":         c["cycle_num"],
            "season_type":       c["season_type"],
            "r_squared":         round(c["r_squared"], 3),
            "cycle_start":       str(cs)[:10],
            "cycle_end":         str(ce)[:10],
            "cycle_length_days": round(c["cycle_length_days"]),
            "gauss_dates":       gauss_dates,
            "gauss_vals":        gauss_vals,
            "sos": {"date": str(ph["sos"])[:10], "val": round(float(pv["sos_ndvi"]), 4)},
            "pos": {"date": str(ph["pos"])[:10], "val": round(float(pv["pos_ndvi"]), 4)},
            "eos": {"date": str(ph["eos"])[:10], "val": round(float(pv["eos_ndvi"]), 4)},
        })

    # Médias fenológicas do hexágono (circular mean de todos os ciclos detectados)
    mean_rows = _pheno[
        (_pheno["id_hexagono"] == hex_id) & (_pheno["cultura"] == cultura)
    ]
    mean_phenophases = [
        {
            "tipo_safra": str(r["tipo_safra"]),
            "sos_doy":   round(float(r["sos_doy"]), 1),
            "pos_doy":   round(float(r["pos_doy"]), 1),
            "eos_doy":   round(float(r["eos_doy"]), 1),
            "n_ciclos":  int(r["n_ciclos"]),
        }
        for _, r in mean_rows.iterrows()
    ]

    out = {
        "hex_id":           hex_id,
        "cultura":          cultura,
        "dates":            dates_str,
        "evi":              [round(float(v), 4) for v in ndvi_arr],
        "smooth":           [round(float(v), 4) for v in ndvi_smooth],
        "cycles":           cycles_out,
        "mean_phenophases": mean_phenophases,
    }
    _cycles_cache[cache_key] = out
    return JSONResponse(out)


if __name__ == "__main__":
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=False)
