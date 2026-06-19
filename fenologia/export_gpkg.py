"""
Exporta fenologia_brasil.parquet como múltiplos GeoPackages para uso no QGIS.

Gera um arquivo .gpkg por combinação cultura+tipo_safra, ex:
  soja_safra.gpkg, soja_safrinha.gpkg, algodao_safra.gpkg, ...

Saída: data/output/gpkg/
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
OUTPUT_DIR = Path("data/output/gpkg")


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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combos = (
        df.groupby(["cultura", "tipo_safra"])
        .size()
        .reset_index(name="n")
        [lambda x: x["n"] >= 5]
    )
    print(f"\nExportando {len(combos)} arquivos .gpkg em {OUTPUT_DIR}/...")

    for _, row in combos.iterrows():
        cultura, tipo = row["cultura"], row["tipo_safra"]
        fname = f"{cultura}_{tipo}.gpkg"
        out_path = OUTPUT_DIR / fname
        out_path.unlink(missing_ok=True)

        sub = df[(df["cultura"] == cultura) & (df["tipo_safra"] == tipo)].copy()
        gdf = build_geodataframe(sub)
        gdf.to_file(out_path, layer=f"{cultura}_{tipo}", driver="GPKG")
        print(f"  {fname}: {len(gdf):,} feições")

    print(f"\nTodos os .gpkg salvos em: {OUTPUT_DIR}/")
    print("No QGIS: arraste qualquer .gpkg para a janela do projeto.")


if __name__ == "__main__":
    main()
