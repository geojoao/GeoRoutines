"""
Exporta fenologia_brasil.parquet como camadas GeoPackage (.gpkg) para uso no QGIS.

Cria dois arquivos:
  - fenologia_brasil.gpkg   : uma camada por cultura+tipo_safra
  - fenologia_brasil_all.gpkg : camada única com todas as combinações

Cada feição é o polígono do hexágono H3 com as colunas fenológicas.
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import h3
except ImportError:
    raise ImportError("pip install h3")

INPUT = Path("data/output/fenologia_brasil.parquet")
OUTPUT_LAYERS = Path("data/output/fenologia_brasil.gpkg")
OUTPUT_ALL = Path("data/output/fenologia_brasil_all.gpkg")

CULTURA_LABELS = {
    "soja": "Soja",
    "cana": "Cana-de-acucar",
    "arroz": "Arroz",
    "algodao": "Algodao",
    "cafe": "Cafe",
    "citrus": "Citrus",
    "dende": "Dende",
    "outras_lavouras_temporarias": "Outras_Temp",
    "outras_lavouras_perenes": "Outras_Peren",
    "segunda_safra": "Segunda_Safra",
    "segunda_safra_algodao": "Segunda_Safra_Algodao",
    "segunda_safra_outras_temporarias": "Segunda_Safra_Temp",
}


def hex_to_polygon(hex_id: str):
    try:
        boundary = h3.cell_to_boundary(hex_id)
        coords = [(lon, lat) for lat, lon in boundary]
        return Polygon(coords)
    except Exception:
        return None


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    print(f"    Construindo geometrias para {len(df):,} hexágonos...", flush=True)
    geoms = [hex_to_polygon(h) for h in df["id_hexagono"]]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    # Arredonda valores numéricos para reduzir tamanho do arquivo
    for col in ["sos_doy", "pos_doy", "eos_doy", "r2_medio", "duracao_media_dias", "evi_maximo"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].round(2)
    return gdf


def export_layers_gpkg(df: pd.DataFrame, output_path: Path):
    """Uma camada por cultura+tipo no mesmo .gpkg."""
    output_path.unlink(missing_ok=True)

    combos = df.groupby(["cultura", "tipo_safra"]).size().reset_index(name="n")
    combos = combos[combos["n"] >= 5]
    print(f"\nExportando {len(combos)} camadas para {output_path.name}...")

    for _, row in combos.iterrows():
        cultura = row["cultura"]
        tipo = row["tipo_safra"]
        layer_name = f"{CULTURA_LABELS.get(cultura, cultura)}_{tipo}"

        sub = df[(df["cultura"] == cultura) & (df["tipo_safra"] == tipo)].copy()
        gdf = build_geodataframe(sub)

        gdf.to_file(output_path, layer=layer_name, driver="GPKG")
        print(f"  Camada '{layer_name}': {len(gdf):,} feições")

    print(f"Salvo: {output_path}")


def export_all_gpkg(df: pd.DataFrame, output_path: Path):
    """Camada única com todas as combinações cultura+tipo."""
    output_path.unlink(missing_ok=True)

    print(f"\nExportando camada única para {output_path.name}...")
    gdf = build_geodataframe(df)
    gdf.to_file(output_path, layer="fenologia_brasil", driver="GPKG")
    print(f"  {len(gdf):,} feições totais")
    print(f"Salvo: {output_path}")


def main():
    print("Carregando fenologia_brasil.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} registros, {df['cultura'].nunique()} culturas, "
          f"{df['tipo_safra'].nunique()} tipos")

    export_layers_gpkg(df, OUTPUT_LAYERS)
    export_all_gpkg(df, OUTPUT_ALL)

    print("\nArquivos prontos para o QGIS:")
    print(f"  {OUTPUT_LAYERS}  — camadas separadas por cultura+safra")
    print(f"  {OUTPUT_ALL}     — camada única com todos os dados")
    print("\nNo QGIS: Camada > Adicionar Camada > Adicionar Camada Vetorial")
    print("  ou arraste o .gpkg para a janela do QGIS.")


if __name__ == "__main__":
    main()
