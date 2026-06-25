"""
Dashboard interativo de fenologia.

Uso (a partir de fenologia/):
    source .venv/bin/activate
    uvicorn dashboard.app:app --port 8000 --reload

Acesse http://localhost:8000
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import h3
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

app = FastAPI(title="Fenologia Brasil")

_pheno: Optional[pd.DataFrame] = None
_ts: Optional[pd.DataFrame] = None    # série temporal indexada por id_hexagono
_grid_cache: dict = {}                # (cultura, tipo) → GeoJSON dict


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
        sub = _pheno[
            (_pheno["cultura"] == cultura) & (_pheno["tipo_safra"] == tipo)
        ]
        features = []
        for _, row in sub.iterrows():
            try:
                bnd    = h3.h3_to_geo_boundary(row["id_hexagono"])
                coords = [[lon, lat] for lat, lon in bnd]
                coords.append(coords[0])   # fecha o polígono
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
        log.info(f"grid {cultura}/{tipo}: {len(features):,} hexágonos gerados")

    return JSONResponse(content=_grid_cache[key])


@app.get("/api/timeseries")
def api_timeseries(hex_id: str, cultura: str):
    col = f"evi_medio_{cultura}"
    if hex_id not in _ts.index:
        raise HTTPException(404, "Hexágono não encontrado na série temporal")

    df = _ts.loc[[hex_id], ["data", col]].dropna(subset=[col]).sort_values("data")

    pheno_rows = _pheno[
        (_pheno["id_hexagono"] == hex_id) & (_pheno["cultura"] == cultura)
    ]
    metrics = [
        {
            "tipo_safra": r["tipo_safra"],
            "sos_doy":    round(float(r["sos_doy"]), 1),
            "pos_doy":    round(float(r["pos_doy"]), 1),
            "eos_doy":    round(float(r["eos_doy"]), 1),
            "n_ciclos":   int(r["n_ciclos"]),
            "knn":        bool(r.get("interpolated", False)),
        }
        for _, r in pheno_rows.iterrows()
    ]

    return {
        "hex_id":  hex_id,
        "dates":   [str(d)[:10] for d in df["data"]],
        "evi":     [round(float(v), 4) for v in df[col]],
        "metrics": metrics,
    }


if __name__ == "__main__":
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=False)
