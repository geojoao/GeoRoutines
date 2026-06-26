"""
Dashboard pixel-level de fenologia — análise de um único hexágono H3.

Carrega o parquet gerado por extract_pixel_series.py e fornece:
  - mapa interativo do hexágono com pixels coloridos por EVI máximo
  - clique em pixel → série EVI + ajuste de logística dupla + extrapolação

Uso (a partir de fenologia/):
    # Primeiro gere o parquet:
    uv run python extract_pixel_series.py 85815527fffffff

    # Depois suba o dashboard:
    PIXEL_HEX=85815527fffffff uvicorn pixel_dashboard.app:app --port 8001 --reload

Acesse http://localhost:8001
"""
from __future__ import annotations

import logging
import os
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

ROOT     = Path(__file__).parent.parent
HTML_PATH = Path(__file__).parent / "index.html"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hexágono alvo (env var ou padrão de teste)
HEX_ID      = os.environ.get("PIXEL_HEX", "85815527fffffff")
PARQUET_DIR = ROOT / "data" / "output" / "pixel_series"
PARQUET_PATH = PARQUET_DIR / f"{HEX_ID}.parquet"

RES_DEG = 1.0 / 240.0   # ~463 m por pixel (meio pixel = RES_DEG/2 por lado)

MIN_CYCLE_DAYS = 80
MAX_CYCLE_DAYS = 180
MIN_EVI_AMPLITUDE = 0.06
MIN_R2            = 0.70

app = FastAPI(title=f"Fenologia Pixel — {HEX_ID}")

# Dados globais
_df: Optional[pd.DataFrame] = None          # série EVI completa (lon, lat, data, evi, soja_ano)
_pixel_summary: Optional[pd.DataFrame] = None  # estatísticas por pixel


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _df, _pixel_summary

    if not PARQUET_PATH.exists():
        log.warning(
            f"Parquet não encontrado: {PARQUET_PATH}\n"
            f"Execute: uv run python extract_pixel_series.py {HEX_ID}"
        )
        return

    log.info(f"Carregando {PARQUET_PATH} ...")
    _df = pd.read_parquet(PARQUET_PATH)
    _df["data"] = pd.to_datetime(_df["data"])
    log.info(f"  {_df.groupby(['lon','lat']).ngroups:,} pixels, {len(_df):,} observações")

    # Estatísticas por pixel para colorir o mapa
    grp = _df.groupby(["lon", "lat"])
    _pixel_summary = grp["evi"].agg(
        evi_max="max",
        evi_mean="mean",
        evi_p90=lambda x: float(np.nanpercentile(x, 90)),
        n_obs="count",
    ).reset_index()

    # Anos em que cada pixel era soja
    soja_years = (
        _df[_df["soja_ano"]]
        .groupby(["lon", "lat"])["data"]
        .apply(lambda s: sorted(s.dt.year.unique().tolist()))
    )
    _pixel_summary = _pixel_summary.merge(
        soja_years.rename("soja_anos").reset_index(),
        on=["lon", "lat"],
        how="left",
    )
    _pixel_summary["soja_anos"] = _pixel_summary["soja_anos"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    log.info("Pronto → http://localhost:8001")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_PATH.read_text("utf-8"))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/info")
def api_info():
    """Metadados do hexágono e do parquet."""
    if _df is None:
        raise HTTPException(503, f"Parquet não carregado. Execute: "
                                 f"uv run python extract_pixel_series.py {HEX_ID}")

    lat_c, lon_c = h3.cell_to_latlng(HEX_ID)
    bnd  = h3.cell_to_boundary(HEX_ID)
    poly = [[lon, lat] for lat, lon in bnd]
    poly.append(poly[0])

    return {
        "hex_id":    HEX_ID,
        "lat":       round(lat_c, 5),
        "lon":       round(lon_c, 5),
        "n_pixels":  int(_df.groupby(["lon", "lat"]).ngroups),
        "n_obs":     len(_df),
        "date_min":  str(_df["data"].min())[:10],
        "date_max":  str(_df["data"].max())[:10],
        "res_deg":   RES_DEG,
        "hex_poly":  poly,
    }


@app.get("/api/pixels")
def api_pixels():
    """
    GeoJSON de pontos (centros de pixel) com estatísticas para colorir o mapa.
    Cada feature é um quadrado de RES_DEG de lado.
    """
    if _pixel_summary is None:
        raise HTTPException(503, "Dados não carregados")

    half = RES_DEG / 2
    features = []
    for _, row in _pixel_summary.iterrows():
        lon, lat = float(row["lon"]), float(row["lat"])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - half, lat - half],
                    [lon + half, lat - half],
                    [lon + half, lat + half],
                    [lon - half, lat + half],
                    [lon - half, lat - half],
                ]],
            },
            "properties": {
                "lon":        lon,
                "lat":        lat,
                "evi_max":    round(float(row["evi_max"]), 4),
                "evi_mean":   round(float(row["evi_mean"]), 4),
                "evi_p90":    round(float(row["evi_p90"]), 4),
                "n_obs":      int(row["n_obs"]),
                "soja_anos":  row["soja_anos"],
            },
        })

    return JSONResponse({"type": "FeatureCollection", "features": features})


@app.get("/api/pixel")
def api_pixel(lon: float, lat: float):
    """
    Série temporal EVI + fenologia ajustada para o pixel (lon, lat).

    Retorna raw EVI, EVI suavizado, ciclos (logística dupla) e extrapolação
    do pico terminal se aplicável — mesmo formato do /api/cycles do dashboard
    de hexágonos.
    """
    if _df is None:
        raise HTTPException(503, "Dados não carregados")

    from fenologia.phenophase import (
        extract_phenometrics, adaptive_smoothing, double_logistic,
        extrapolate_terminal_cycle,
    )

    # Tolerância de meio pixel para achar o pixel mais próximo
    tol = RES_DEG * 0.6
    mask = (np.abs(_df["lon"] - lon) < tol) & (np.abs(_df["lat"] - lat) < tol)
    sub = _df[mask].sort_values("data")

    if len(sub) < 10:
        raise HTTPException(404, f"Pixel ({lon:.5f}, {lat:.5f}) não encontrado ou série curta")

    ts_df = sub[["data", "evi"]].rename(columns={"data": "datetime", "evi": "NDVI_mean"}).copy()
    ndvi_arr  = ts_df["NDVI_mean"].values.astype(float)
    dates_arr = ts_df["datetime"].values

    result = extract_phenometrics(
        ts_df,
        ndvi_column="NDVI_mean",
        min_cycle_length_days=MIN_CYCLE_DAYS,
        smoothing_method="both",
        quality_threshold=MIN_R2,
    )

    ndvi_smooth = adaptive_smoothing(ndvi_arr, dates_arr, method="both")

    # Ciclos com logística dupla
    cycles_out = []
    for c in result.get("cycles", []):
        if not c.get("fit_success"):
            continue
        growing = c["phenophase_days"]["eos_days"] - c["phenophase_days"]["sos_days"]
        amp = c["curve_params"]["amplitude"]
        r2  = c["r_squared"]
        if not (20 <= growing <= MAX_CYCLE_DAYS):
            continue
        if amp < MIN_EVI_AMPLITUDE or r2 < MIN_R2:
            continue

        cp = c["curve_params"]
        ph = c["phenophase_dates"]
        pv = c["phenophase_values"]
        cs = pd.Timestamp(c["cycle_start"])
        ce = pd.Timestamp(c["cycle_end"])
        n_pts = max(60, int(c["cycle_length_days"] / 3))
        curve_dates, curve_vals = [], []
        for i in range(n_pts + 1):
            t = cs + pd.Timedelta(days=i * c["cycle_length_days"] / n_pts)
            if t > ce:
                break
            x = (t - cs).total_seconds() / 86400
            y = float(double_logistic(
                np.array([x]),
                cp["amplitude"], cp["m1"], cp["k1"], cp["m2"], cp["k2"], cp["offset"]
            )[0])
            curve_dates.append(str(t)[:10])
            curve_vals.append(round(y, 4))

        cycles_out.append({
            "cycle_num":         c["cycle_num"],
            "season_type":       c.get("season_type"),
            "r_squared":         round(r2, 3),
            "cycle_start":       str(cs)[:10],
            "cycle_end":         str(ce)[:10],
            "cycle_length_days": round(c["cycle_length_days"]),
            "curve_dates":       curve_dates,
            "curve_vals":        curve_vals,
            "sos": {"date": str(ph["sos"])[:10], "val": round(float(pv["sos_ndvi"]), 4)},
            "pos": {"date": str(ph["pos"])[:10], "val": round(float(pv["pos_ndvi"]), 4)},
            "eos": {"date": str(ph["eos"])[:10], "val": round(float(pv["eos_ndvi"]), 4)},
        })

    # Extrapolação do pico terminal
    extrap_raw = extrapolate_terminal_cycle(ndvi_smooth, dates_arr, result.get("cycles", []))
    if extrap_raw.get("success"):
        cv = extrap_raw["curve"]
        fc_idx = [i for i, f in enumerate(cv["is_forecast"]) if f]
        extrap_out: dict = {
            "success":           True,
            "curve_dates":       [cv["dates"][i]        for i in fc_idx],
            "curve_vals":        [cv["values"][i]       for i in fc_idx],
            "curve_upper":       [cv["values_upper"][i] for i in fc_idx],
            "curve_lower":       [cv["values_lower"][i] for i in fc_idx],
            "forecast_eos_date": str(extrap_raw["forecast_eos_date"])[:10],
            "series_end_date":   str(extrap_raw["series_end_date"])[:10],
            "uncertainty_days":  extrap_raw["uncertainty_days"],
            "peak_was_observed": extrap_raw["peak_was_observed"],
            "n_prior_cycles":    extrap_raw["n_prior_cycles"],
        }
    else:
        extrap_out = {"success": False, "reason": extrap_raw.get("reason", "")}

    return JSONResponse({
        "lon":    lon,
        "lat":    lat,
        "dates":  [str(d)[:10] for d in sub["data"]],
        "evi":    [round(float(v), 4) for v in ndvi_arr],
        "smooth": [round(float(v), 4) for v in ndvi_smooth],
        "cycles": cycles_out,
        "extrap": extrap_out,
        "soja_anos": sorted(_df[mask]["data"].dt.year[_df[mask]["soja_ano"]].unique().tolist()),
    })


if __name__ == "__main__":
    uvicorn.run("pixel_dashboard.app:app", host="0.0.0.0", port=8001, reload=False)
