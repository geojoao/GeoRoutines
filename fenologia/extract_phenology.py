"""
Extrai métricas fenológicas (SOS/plantio, EOS/colheita) por hexágono e cultura
a partir dos arquivos _parts/, usando o módulo phenophase.py.

Cada ciclo detectado é classificado como 'safra' ou 'safrinha' pela data do POS:
  - Safra:    POS em out–mar (DOY ≤ 90 ou ≥ 274) — início das chuvas
  - Safrinha: POS em abr–set (DOY 91–273)         — meio do ano seco

Saída: data/output/fenologia_brasil.parquet
  Colunas: id_hexagono, cultura, tipo_safra, n_ciclos, sos_doy, pos_doy, eos_doy,
            r2_medio, duracao_media_dias, evi_maximo
"""

import sys
import numpy as np
import pandas as pd
import warnings
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

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

PARTS_DIR = Path("data/output/_parts")
OUTPUT = Path("data/output/fenologia_brasil.parquet")
N_WORKERS = 14


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

    evi = evi.astype(float)
    mask = ~np.isnan(evi)
    if mask.sum() < 10:
        return []

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
        return []

    if not result["success"]:
        return []

    successful = [
        c for c in result["cycles"]
        if c.get("fit_success")
        and c["cycle_length_days"] >= min_days
        and c["gaussian_params"]["amplitude"] >= MIN_EVI_AMPLITUDE
    ]

    if not successful:
        return []

    evi_max = float(sub["NDVI_mean"].max())

    # Agrupa ciclos por tipo_safra (safra / safrinha)
    by_season = defaultdict(list)
    for c in successful:
        season = c.get("season_type", "safra")
        by_season[season].append(c)

    records = []
    for season, cycles in by_season.items():
        sos_doys = [c["phenophase_dates"]["sos"].day_of_year for c in cycles]
        pos_doys = [c["phenophase_dates"]["pos"].day_of_year for c in cycles]
        eos_doys = [c["phenophase_dates"]["eos"].day_of_year for c in cycles]
        r2s = [c["r_squared"] for c in cycles]
        lengths = [c["cycle_length_days"] for c in cycles]

        records.append({
            "id_hexagono": hex_id,
            "cultura": cultura,
            "tipo_safra": season,
            "n_ciclos": len(cycles),
            "sos_doy": circular_mean_doy(sos_doys),
            "pos_doy": circular_mean_doy(pos_doys),
            "eos_doy": circular_mean_doy(eos_doys),
            "r2_medio": float(np.mean(r2s)),
            "duracao_media_dias": float(np.mean(lengths)),
            "evi_maximo": evi_max,
        })

    return records


def load_evi_from_parts() -> pd.DataFrame:
    """Concatena todos os arquivos _parts/ para reconstruir a série temporal."""
    parts = sorted(PARTS_DIR.glob("*.parquet"))
    print(f"  Carregando {len(parts)} arquivos _parts/...", flush=True)
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    # Remove duplicatas (hexagono+data) que podem existir se tiles se sobrepõem
    df = df.drop_duplicates(subset=["id_hexagono", "data"])
    print(f"  {len(df):,} linhas, {df['id_hexagono'].nunique():,} hexágonos únicos", flush=True)
    return df


def process_cultura(df: pd.DataFrame, cultura: str) -> list[dict]:
    col = f"evi_medio_{cultura}"
    if col not in df.columns:
        print(f"  [{cultura}] coluna {col} não encontrada, pulando.")
        return []
    sub = df[["id_hexagono", "data", col]].dropna(subset=[col])
    hex_ids = sub["id_hexagono"].unique()
    print(f"  [{cultura}] {len(hex_ids):,} hexágonos...", flush=True)

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

    # Cada resultado é uma lista (pode ser vazia)
    valid = [r for result in results for r in result]
    print(f"  [{cultura}] -> {len(valid):,} registros (safra+safrinha)", flush=True)
    return valid


def main():
    print("Carregando séries temporais EVI dos _parts/...")
    df = load_evi_from_parts()

    all_records = []
    for cultura in CULTURAS:
        records = process_cultura(df, cultura)
        all_records.extend(records)

    print(f"\nTotal de registros: {len(all_records):,}")
    out = pd.DataFrame(all_records)
    out.to_parquet(OUTPUT, index=False)
    print(f"Salvo em {OUTPUT}")
    summary = out.groupby(["cultura", "tipo_safra"])[
        ["n_ciclos", "r2_medio", "duracao_media_dias"]
    ].mean().round(2)
    print(summary)


if __name__ == "__main__":
    main()
