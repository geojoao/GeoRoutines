"""
Exporta fenologia_brasil.parquet como GeoPackage (.gpkg) para uso no QGIS.

Gera fenologia_brasil.gpkg com uma camada por cultura (12 camadas).
Cada camada contém safra e safrinha juntas, diferenciadas pela coluna tipo_safra.
"""

import warnings
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
OUTPUT = Path("data/output/fenologia_brasil.gpkg")

CULTURA_LABELS = {
    "soja": "Soja",
    "cana": "Cana_de_Acucar",
    "arroz": "Arroz",
    "algodao": "Algodao",
    "cafe": "Cafe",
    "citrus": "Citrus",
    "dende": "Dende",
    "outras_lavouras_temporarias": "Outras_Temporarias",
    "outras_lavouras_perenes": "Outras_Perenes",
    "segunda_safra": "Segunda_Safra",
    "segunda_safra_algodao": "Segunda_Safra_Algodao",
    "segunda_safra_outras_temporarias": "Segunda_Safra_Outras_Temp",
}


def hex_to_polygon(hex_id: str):
    try:
        boundary = h3.cell_to_boundary(hex_id)
        return Polygon([(lon, lat) for lat, lon in boundary])
    except Exception:
        return None


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geoms = [hex_to_polygon(h) for h in df["id_hexagono"]]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    for col in ["sos_doy", "pos_doy", "eos_doy", "r2_medio", "duracao_media_dias", "evi_maximo"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].round(2)
    return gdf


def main():
    print("Carregando fenologia_brasil.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} registros, {df['cultura'].nunique()} culturas")

    OUTPUT.unlink(missing_ok=True)

    culturas = sorted(df["cultura"].unique())
    print(f"\nExportando {len(culturas)} camadas para {OUTPUT.name}...")

    for cultura in culturas:
        layer_name = CULTURA_LABELS.get(cultura, cultura)
        sub = df[df["cultura"] == cultura].copy()
        print(f"  [{layer_name}] {len(sub):,} feições (safra+safrinha)...", flush=True)
        gdf = build_geodataframe(sub)
        gdf.to_file(OUTPUT, layer=layer_name, driver="GPKG")

    print(f"\nSalvo: {OUTPUT}")
    print(f"  {len(culturas)} camadas — uma por cultura, com coluna tipo_safra para filtrar safra/safrinha")
    print("\nNo QGIS: arraste o .gpkg para a janela do projeto")
    print("  ou Camada → Adicionar Camada → Adicionar Camada Vetorial")


if __name__ == "__main__":
    main()
