"""
interpolate_phenology.py

Pós-processamento da fenologia em dois passos:

1. Filtro de qualidade: descarta hexágonos com n_ciclos < 3
   (menos de 3 anos detectados nos 5 anos da série = sinal inconsistente)

2. Interpolação KNN espacial: preenche hexágonos vazios ENTRE os dados
   reais usando média circular ponderada por distância H3.
   Limite: apenas hexágonos dentro de MAX_DISTANCE células do vizinho
   mais próximo com dado real.

DOY é circular (365→1), então usa senos/cossenos para evitar artefatos
na virada do ano (ex: média de 355 e 10 = 182 errado → correto = ~2.5).

Saída:
  data/output/fenologia_brasil_knn.parquet
  data/output/gpkg_knn/<cultura>_<tipo>.gpkg  (24 arquivos)

Coluna `interpolated`:
  False = dado real (n_ciclos >= MIN_CYCLES)
  True  = preenchido por KNN
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

try:
    import h3
except ImportError:
    raise ImportError("pip install h3")

INPUT          = Path("data/output/fenologia_brasil.parquet")
OUTPUT_PARQUET = Path("data/output/fenologia_brasil_knn.parquet")
OUTPUT_GPKG_DIR = Path("data/output/gpkg_knn")

MIN_CYCLES   = 3
MAX_DISTANCE = 3   # células H3
DOY_COLS     = ["sos_doy", "pos_doy", "eos_doy"]
METRIC_COLS  = ["r2_medio", "duracao_media_dias", "evi_maximo"]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def circular_weighted_mean_doy(doys: list, weights: list) -> float:
    """Média circular ponderada de valores DOY (0–365, cíclico)."""
    angles = np.array(doys, dtype=float) * 2 * np.pi / 365.25
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    mean_angle = np.arctan2(np.sum(w * np.sin(angles)),
                            np.sum(w * np.cos(angles)))
    return float(mean_angle * 365.25 / (2 * np.pi) % 365.25)


def hex_to_polygon(hex_id: str):
    try:
        boundary = h3.cell_to_boundary(hex_id)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Interpolação KNN
# ---------------------------------------------------------------------------

def interpolate_cultura_tipo(sub: pd.DataFrame,
                              cultura: str, tipo: str) -> list[dict]:
    """
    Para cada hexágono vazio dentro de MAX_DISTANCE células dos dados reais,
    interpola sos/pos/eos_doy com média circular ponderada por 1/distância.

    Usa grid_ring(d) ao expandir a partir de cada dado para montar o mapa
    candidato→vizinhos sem recalcular grid_distance individualmente.
    """
    data_hexes = set(sub["id_hexagono"])
    lookup = (sub.set_index("id_hexagono")[DOY_COLS + METRIC_COLS]
                 .to_dict("index"))

    # candidato → [(data_hex, distance), ...]
    candidate_neighbors: dict[str, list] = defaultdict(list)

    for data_hex in data_hexes:
        for d in range(1, MAX_DISTANCE + 1):
            try:
                ring = h3.grid_ring(data_hex, d)
            except Exception:
                continue
            for candidate in ring:
                if candidate not in data_hexes:
                    candidate_neighbors[candidate].append((data_hex, d))

    print(f"    {len(data_hexes):,} hexs reais → "
          f"{len(candidate_neighbors):,} candidatos KNN", flush=True)

    filled = []
    for candidate, neighbors in candidate_neighbors.items():
        hexes   = [h for h, _ in neighbors]
        weights = [1.0 / d for _, d in neighbors]  # peso = 1/distância

        row: dict = {
            "id_hexagono": candidate,
            "cultura":     cultura,
            "tipo_safra":  tipo,
            "n_ciclos":    0,
            "interpolated": True,
        }

        for col in DOY_COLS:
            row[col] = circular_weighted_mean_doy(
                [lookup[h][col] for h in hexes], weights
            )

        for col in METRIC_COLS:
            row[col] = float(np.average([lookup[h][col] for h in hexes],
                                         weights=weights))

        filled.append(row)

    return filled


# ---------------------------------------------------------------------------
# Exportação GPKG
# ---------------------------------------------------------------------------

def export_gpkg(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for (cultura, tipo), sub in df.groupby(["cultura", "tipo_safra"]):
        geoms = [hex_to_polygon(h) for h in sub["id_hexagono"]]
        gdf = gpd.GeoDataFrame(sub.copy(), geometry=geoms, crs="EPSG:4326")
        gdf = gdf[gdf.geometry.notna()]

        for col in DOY_COLS + METRIC_COLS:
            if col in gdf.columns:
                gdf[col] = gdf[col].round(2)

        out_path = output_dir / f"{cultura}_{tipo}.gpkg"
        out_path.unlink(missing_ok=True)
        gdf.to_file(out_path, layer=f"{cultura}_{tipo}", driver="GPKG")

        n_real = int((sub["interpolated"] == False).sum())
        n_knn  = int((sub["interpolated"] == True).sum())
        print(f"  {cultura}_{tipo}.gpkg: "
              f"{n_real:,} reais + {n_knn:,} KNN = {len(gdf):,} total")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Carregando fenologia_brasil.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} registros totais")

    # --- Filtro de qualidade ---
    df_ok = df[df["n_ciclos"] >= MIN_CYCLES].copy()
    df_ok["interpolated"] = False
    n_drop = len(df) - len(df_ok)
    print(f"  Filtro n_ciclos >= {MIN_CYCLES}: "
          f"{len(df_ok):,} mantidos, {n_drop:,} descartados "
          f"({n_drop / len(df) * 100:.1f}%)")

    # --- Interpolação KNN ---
    print(f"\nInterpolando (MAX_DISTANCE={MAX_DISTANCE} células H3)...")
    all_filled = []
    for (cultura, tipo), sub in df_ok.groupby(["cultura", "tipo_safra"]):
        print(f"  [{cultura}/{tipo}]", flush=True)
        filled = interpolate_cultura_tipo(sub, cultura, tipo)
        print(f"    → {len(filled):,} hexágonos interpolados")
        all_filled.extend(filled)

    df_knn = pd.DataFrame(all_filled) if all_filled else pd.DataFrame()

    result = pd.concat([df_ok, df_knn], ignore_index=True)

    print(f"\nTotal final: {len(result):,} registros")
    print(f"  Reais:        {len(df_ok):,}")
    print(f"  Interpolados: {len(df_knn):,}")

    result.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Parquet salvo: {OUTPUT_PARQUET}")

    print(f"\nExportando GPKGs para {OUTPUT_GPKG_DIR}/...")
    export_gpkg(result, OUTPUT_GPKG_DIR)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
