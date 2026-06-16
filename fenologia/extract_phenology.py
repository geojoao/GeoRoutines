"""
Extrai métricas fenológicas (SOS/plantio, EOS/colheita) por hexágono e cultura
a partir do evi_brazil.parquet, usando o módulo phenophase.py.

Saída: data/output/fenologia_brasil.parquet
"""

import sys
import numpy as np
import pandas as pd
import warnings
import multiprocessing as mp
from pathlib import Path
from functools import partial

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from fenologia.phenophase import extract_phenometrics

CULTURAS = [
    "soja",
    "cana",
    "arroz",
    "algodao",
    "cafe",
    "citrus",
    "dende",
    "outras_lavouras_temporarias",
    "outras_lavouras_perenes",
    "segunda_safra",
    "segunda_safra_algodao",
    "segunda_safra_outras_temporarias",
]

MIN_CYCLE_DAYS = {
    "soja": 70,
    "cana": 120,
    "arroz": 60,
    "algodao": 70,
    "cafe": 90,
    "citrus": 90,
    "dende": 90,
    "outras_lavouras_temporarias": 60,
    "outras_lavouras_perenes": 90,
    "segunda_safra": 45,
    "segunda_safra_algodao": 45,
    "segunda_safra_outras_temporarias": 45,
}

MIN_EVI_AMPLITUDE = 0.08

INPUT = Path("data/output/evi_brazil.parquet")
OUTPUT = Path("data/output/fenologia_brasil.parquet")
N_WORKERS = 14  # deixa 2 livres no sistema


def circular_mean_doy(doys):
    rad = np.deg2rad(np.array(doys, dtype=float) * 360.0 / 365.25)
    mean_sin = np.nanmean(np.sin(rad))
    mean_cos = np.nanmean(np.cos(rad))
    mean_rad = np.arctan2(mean_sin, mean_cos)
    doy = np.rad2deg(mean_rad) * 365.25 / 360.0
    return float(doy % 365.25)


def _worker(args):
    """Processado em worker separado — args = (hex_id, dates_array, evi_array, cultura)."""
    warnings.filterwarnings("ignore")
    hex_id, dates, evi, cultura = args

    mask = ~np.isnan(evi)
    if mask.sum() < 10:
        return None

    sub = pd.DataFrame({"datetime": dates[mask], "NDVI_mean": evi[mask]})
    min_days = MIN_CYCLE_DAYS.get(cultura, 60)

    try:
        result = extract_phenometrics(
            sub,
            ndvi_column="NDVI_mean",
            min_cycle_length_days=min_days,
            smoothing_method="both",
            quality_threshold=0.5,
        )
    except Exception:
        return None

    if not result["success"]:
        return None

    successful = [
        c for c in result["cycles"]
        if c.get("fit_success")
        and c["cycle_length_days"] >= min_days
        and c["gaussian_params"]["amplitude"] >= MIN_EVI_AMPLITUDE
    ]

    if not successful:
        return None

    sos_doys = [c["phenophase_dates"]["sos"].day_of_year for c in successful]
    pos_doys = [c["phenophase_dates"]["pos"].day_of_year for c in successful]
    eos_doys = [c["phenophase_dates"]["eos"].day_of_year for c in successful]
    r2s = [c["r_squared"] for c in successful]
    lengths = [c["cycle_length_days"] for c in successful]

    return {
        "id_hexagono": hex_id,
        "cultura": cultura,
        "n_ciclos": len(successful),
        "sos_doy": circular_mean_doy(sos_doys),
        "pos_doy": circular_mean_doy(pos_doys),
        "eos_doy": circular_mean_doy(eos_doys),
        "r2_medio": float(np.mean(r2s)),
        "duracao_media_dias": float(np.mean(lengths)),
        "evi_maximo": float(sub["NDVI_mean"].max()),
    }


def process_cultura(df: pd.DataFrame, cultura: str) -> list[dict]:
    col = f"evi_medio_{cultura}"
    sub = df[["id_hexagono", "data", col]].dropna(subset=[col])
    hex_ids = sub["id_hexagono"].unique()
    print(f"  [{cultura}] {len(hex_ids):,} hexágonos...", flush=True)

    # Monta lista de argumentos (arrays numpy são mais baratos para serialização)
    grouped = sub.groupby("id_hexagono")
    tasks = []
    for hex_id, grp in grouped:
        grp_sorted = grp.sort_values("data")
        tasks.append((
            hex_id,
            grp_sorted["data"].values,
            grp_sorted[col].values,
            cultura,
        ))

    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(_worker, tasks, chunksize=20)

    valid = [r for r in results if r is not None]
    print(f"  [{cultura}] -> {len(valid):,} ajustes válidos", flush=True)
    return valid


def main():
    print("Carregando evi_brazil.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} linhas, {df['id_hexagono'].nunique():,} hexágonos")

    all_records = []
    for cultura in CULTURAS:
        records = process_cultura(df, cultura)
        all_records.extend(records)

    print(f"\nTotal de registros: {len(all_records):,}")
    out = pd.DataFrame(all_records)
    out.to_parquet(OUTPUT, index=False)
    print(f"Salvo em {OUTPUT}")
    print(out.groupby("cultura")[["n_ciclos", "r2_medio", "duracao_media_dias"]].mean().round(2))


if __name__ == "__main__":
    main()
