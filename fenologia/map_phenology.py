"""
Gera mapas hexagonais do Brasil coloridos por DOY de plantio (SOS) e colheita (EOS)
para cada cultura.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colorbar import ColorbarBase
from shapely.geometry import Polygon
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import h3
    H3_VERSION = 3
except ImportError:
    raise ImportError("pip install h3")

INPUT = Path("data/output/fenologia_brasil.parquet")
OUTPUT_DIR = Path("data/output/mapas_fenologia")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Culturas com labels amigáveis
CULTURA_LABELS = {
    "soja": "Soja",
    "cana": "Cana-de-açúcar",
    "arroz": "Arroz",
    "algodao": "Algodão",
    "cafe": "Café",
    "citrus": "Citrus",
    "dende": "Dendê",
    "outras_lavouras_temporarias": "Outras Lavouras Temporárias",
    "outras_lavouras_perenes": "Outras Lavouras Perenes",
    "segunda_safra": "Segunda Safra",
    "segunda_safra_algodao": "Segunda Safra Algodão",
    "segunda_safra_outras_temporarias": "Segunda Safra Outras Temporárias",
}

MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
MES_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


def doy_to_date_str(doy: float) -> str:
    """Converte DOY para string de mês."""
    doy = int(doy) % 365
    for i, md in enumerate(MES_DOYS):
        if i < 11 and doy < MES_DOYS[i + 1]:
            return MESES[i]
    return MESES[11]


def hex_to_polygon(hex_id: str):
    """Converte ID H3 para polígono Shapely."""
    try:
        boundary = h3.cell_to_boundary(hex_id)
        coords = [(lon, lat) for lat, lon in boundary]
        return Polygon(coords)
    except Exception:
        return None


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Constrói GeoDataFrame com geometrias H3."""
    print("  Construindo geometrias hexagonais...", flush=True)
    geoms = [hex_to_polygon(h) for h in df["id_hexagono"]]
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()]
    return gdf


def get_brazil_outline() -> gpd.GeoDataFrame:
    """Carrega contorno do Brasil do Natural Earth."""
    ne_path = "/scratch/opt-coiled/cache/uv/archive-v0/r3ZF462tIwZUe9Hv/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"
    world = gpd.read_file(ne_path)
    return world[world["name"] == "Brazil"]


def make_doy_colormap():
    """Colormap cíclico para DOY (começa e termina no mesmo ponto)."""
    return plt.cm.twilight_shifted


def plot_phenology_map(gdf: gpd.GeoDataFrame, doy_col: str, cultura: str,
                       metric_label: str, brazil: gpd.GeoDataFrame,
                       ax: plt.Axes, cmap):
    """Plota um mapa fenológico em um eixo."""
    vmin, vmax = 1, 365

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    gdf.plot(
        column=doy_col,
        ax=ax,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        alpha=0.85,
    )
    brazil.boundary.plot(ax=ax, color="black", linewidth=0.6)

    ax.set_xlim(-74, -34)
    ax.set_ylim(-34, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{CULTURA_LABELS.get(cultura, cultura)}\n{metric_label}",
        fontsize=11, fontweight="bold", pad=4,
    )

    return norm


def add_doy_colorbar(fig, ax_cb, norm, cmap):
    """Adiciona colorbar com ticks em meses."""
    cb = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks(MES_DOYS)
    cb.set_ticklabels(MESES)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Dia do ano (DOY)", fontsize=9)


def plot_cultura(df: pd.DataFrame, cultura: str, brazil: gpd.GeoDataFrame):
    """Gera figura com mapas de SOS e EOS para uma cultura."""
    sub = df[df["cultura"] == cultura].copy()
    if len(sub) < 10:
        print(f"  Poucos dados para {cultura}, pulando.")
        return

    print(f"  Mapa: {cultura} ({len(sub)} hexágonos)", flush=True)
    gdf = build_geodataframe(sub)

    cmap = make_doy_colormap()

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"Fenologia — {CULTURA_LABELS.get(cultura, cultura)} (Brasil 2020–2024)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Layout: 2 mapas + 1 colorbar
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.02)
    ax_sos = fig.add_subplot(gs[0])
    ax_eos = fig.add_subplot(gs[1])
    ax_cb = fig.add_subplot(gs[2])

    norm = plot_phenology_map(gdf, "sos_doy", cultura, "Plantio (SOS)", brazil, ax_sos, cmap)
    plot_phenology_map(gdf, "eos_doy", cultura, "Colheita (EOS)", brazil, ax_eos, cmap)
    add_doy_colorbar(fig, ax_cb, norm, cmap)

    # Estatísticas rápidas
    sos_label = doy_to_date_str(sub["sos_doy"].median())
    eos_label = doy_to_date_str(sub["eos_doy"].median())
    fig.text(0.5, -0.02,
             f"Mediana: Plantio ≈ {sos_label}  |  Colheita ≈ {eos_label}  |  "
             f"R² médio: {sub['r2_medio'].mean():.2f}  |  "
             f"Hexágonos: {len(gdf):,}",
             ha="center", fontsize=9, color="#444444")

    out_path = OUTPUT_DIR / f"fenologia_{cultura}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Salvo: {out_path}")


def plot_overview(df: pd.DataFrame, brazil: gpd.GeoDataFrame, culturas_foco: list[str]):
    """Painel geral com SOS das principais culturas."""
    cmap = make_doy_colormap()
    n = len(culturas_foco)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("Mapa de Plantio (SOS) por Cultura — Brasil 2020–2024",
                 fontsize=16, fontweight="bold", y=1.01)

    norm = mcolors.Normalize(vmin=1, vmax=365)

    for i, cultura in enumerate(culturas_foco):
        ax = axes[i]
        sub = df[df["cultura"] == cultura].copy()
        if len(sub) < 10:
            ax.axis("off")
            continue
        gdf = build_geodataframe(sub)
        plot_phenology_map(gdf, "sos_doy", cultura, "Plantio (SOS)", brazil, ax, cmap)

    # Remove eixos extras
    for j in range(len(culturas_foco), len(axes)):
        axes[j].axis("off")

    # Colorbar compartilhado
    cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    add_doy_colorbar(fig, cb_ax, norm, cmap)

    out_path = OUTPUT_DIR / "fenologia_painel_plantio.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Painel SOS salvo: {out_path}")


def main():
    print("Carregando fenologia_brasil.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} registros, {df['cultura'].nunique()} culturas")
    print(f"  Culturas: {sorted(df['cultura'].unique())}")

    print("\nCarregando contorno do Brasil...")
    brazil = get_brazil_outline()

    culturas_principais = ["soja", "cana", "algodao", "cafe", "citrus", "arroz", "dende",
                           "segunda_safra"]

    # Mapas individuais por cultura
    print("\nGerando mapas individuais...")
    for cultura in df["cultura"].unique():
        plot_cultura(df, cultura, brazil)

    # Painel resumo
    print("\nGerando painel geral de plantio...")
    plot_overview(df, brazil, culturas_principais)

    print(f"\nTodos os mapas salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
