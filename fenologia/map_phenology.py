"""
Gera mapas hexagonais do Brasil coloridos por DOY de plantio (SOS) e colheita (EOS)
para cada cultura, separando safra e safrinha.
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
except ImportError:
    raise ImportError("pip install h3")

INPUT = Path("data/output/fenologia_brasil.parquet")
OUTPUT_DIR = Path("data/output/mapas_fenologia")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

TIPO_LABELS = {
    "safra": "Safra (estação principal — out–mar)",
    "safrinha": "Safrinha (segunda safra — abr–set)",
}

MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
MES_DOYS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]


def doy_to_date_str(doy: float) -> str:
    doy = int(doy) % 365
    for i, md in enumerate(MES_DOYS):
        if i < 11 and doy < MES_DOYS[i + 1]:
            return MESES[i]
    return MESES[11]


def hex_to_polygon(hex_id: str):
    try:
        boundary = h3.cell_to_boundary(hex_id)
        coords = [(lon, lat) for lat, lon in boundary]
        return Polygon(coords)
    except Exception:
        return None


def build_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geoms = [hex_to_polygon(h) for h in df["id_hexagono"]]
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()]
    return gdf


def get_brazil_outline() -> gpd.GeoDataFrame:
    import importlib.resources
    import pyogrio
    # Localiza naturalearth_lowres dentro do pacote pyogrio instalado
    pyogrio_dir = Path(pyogrio.__file__).parent
    ne_path = pyogrio_dir / "tests" / "fixtures" / "naturalearth_lowres" / "naturalearth_lowres.shp"
    world = gpd.read_file(ne_path)
    return world[world["name"] == "Brazil"]


def _monthly_cmap_norm(series: pd.Series, _unused=None):
    """Colormap twilight_shifted contínuo com normalização 1–365 e ticks mensais."""
    cmap = plt.cm.twilight_shifted
    norm = mcolors.Normalize(vmin=1, vmax=365)
    tick_positions = MES_DOYS
    tick_labels = MESES
    return cmap, norm, tick_positions, tick_labels


def plot_single_map(gdf: gpd.GeoDataFrame, doy_col: str, title: str,
                    subtitle: str, brazil: gpd.GeoDataFrame,
                    median_label: str, n_hex: int, r2: float,
                    out_path: Path, base_cmap=None):
    """Gera mapa com twilight_shifted cíclico (azul→branco→vermelho→preto)."""
    cmap, norm, tick_pos, tick_labels = _monthly_cmap_norm(gdf[doy_col])

    fig = plt.figure(figsize=(9, 10))
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", y=1.005)

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.055], wspace=0.03)
    ax_map = fig.add_subplot(gs[0])
    ax_cb  = fig.add_subplot(gs[1])

    gdf.plot(column=doy_col, ax=ax_map, cmap=cmap, norm=norm,
             linewidth=0, alpha=0.9)
    brazil.boundary.plot(ax=ax_map, color="black", linewidth=0.7)
    ax_map.set_xlim(-74, -34)
    ax_map.set_ylim(-34, 6)
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    cb = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="vertical",
                      spacing="proportional")
    cb.set_ticks(tick_pos)
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Mês", fontsize=10)

    fig.text(
        0.5, -0.01,
        f"Mediana ≈ {median_label}  |  R² médio: {r2:.2f}  |  {n_hex:,} hexágonos",
        ha="center", fontsize=9, color="#444444",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Salvo: {out_path}")


def plot_cultura_tipo(df: pd.DataFrame, cultura: str, tipo: str,
                      brazil: gpd.GeoDataFrame):
    """Gera mapa separado de plantio (SOS) e colheita (EOS)."""
    sub = df[(df["cultura"] == cultura) & (df["tipo_safra"] == tipo)].copy()
    if len(sub) < 10:
        print(f"  Poucos dados para {cultura}/{tipo} ({len(sub)}), pulando.")
        return

    cultura_label = CULTURA_LABELS.get(cultura, cultura)
    tipo_label = TIPO_LABELS.get(tipo, tipo)
    print(f"  Mapa: {cultura}/{tipo} ({len(sub)} hexágonos)", flush=True)

    gdf = build_geodataframe(sub)
    r2 = sub["r2_medio"].mean()
    n_hex = len(gdf)

    plot_single_map(
        gdf, "sos_doy",
        title=f"{cultura_label} — Plantio (SOS)",
        subtitle=f"{tipo_label} | Brasil 2020–2024",
        brazil=brazil,
        median_label=doy_to_date_str(sub["sos_doy"].median()),
        n_hex=n_hex, r2=r2,
        out_path=OUTPUT_DIR / f"fenologia_{cultura}_{tipo}_plantio.png",
        base_cmap=None,
    )

    plot_single_map(
        gdf, "eos_doy",
        title=f"{cultura_label} — Colheita (EOS)",
        subtitle=f"{tipo_label} | Brasil 2020–2024",
        brazil=brazil,
        median_label=doy_to_date_str(sub["eos_doy"].median()),
        n_hex=n_hex, r2=r2,
        out_path=OUTPUT_DIR / f"fenologia_{cultura}_{tipo}_colheita.png",
        base_cmap=None,
    )


def plot_overview(df: pd.DataFrame, brazil: gpd.GeoDataFrame,
                  culturas_foco: list, tipo: str):
    """Painel geral com SOS de um tipo (safra ou safrinha)."""
    sub_tipo = df[df["tipo_safra"] == tipo]
    culturas_com_dados = [c for c in culturas_foco
                          if len(sub_tipo[sub_tipo["cultura"] == c]) >= 10]
    if not culturas_com_dados:
        return

    tipo_label = TIPO_LABELS.get(tipo, tipo)
    cmap = plt.cm.twilight_shifted
    n = len(culturas_com_dados)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"Mapa de Plantio (SOS) — {tipo_label} | Brasil 2020–2024",
        fontsize=14, fontweight="bold", y=1.01,
    )

    norm = mcolors.Normalize(vmin=1, vmax=365)

    for i, cultura in enumerate(culturas_com_dados):
        ax = axes[i]
        sub = sub_tipo[sub_tipo["cultura"] == cultura].copy()
        gdf = build_geodataframe(sub)
        gdf.plot(column="sos_doy", ax=ax, cmap=cmap, norm=norm, linewidth=0, alpha=0.85)
        brazil.boundary.plot(ax=ax, color="black", linewidth=0.6)
        ax.set_xlim(-74, -34)
        ax.set_ylim(-34, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{CULTURA_LABELS.get(cultura, cultura)}\nPlantio (SOS)",
                     fontsize=10, fontweight="bold", pad=4)

    for j in range(len(culturas_com_dados), len(axes)):
        axes[j].axis("off")

    cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = ColorbarBase(cb_ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks(MES_DOYS)
    cb.set_ticklabels(MESES)
    cb_ax.tick_params(labelsize=8)
    cb.set_label("Dia do ano (DOY)", fontsize=9)

    out_path = OUTPUT_DIR / f"fenologia_painel_plantio_{tipo}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Painel SOS ({tipo}) salvo: {out_path}")


def main():
    print("Carregando fenologia_brasil.parquet...")
    df = pd.read_parquet(INPUT)
    print(f"  {len(df):,} registros, {df['cultura'].nunique()} culturas")
    print(f"  Tipos: {sorted(df['tipo_safra'].unique())}")
    print(f"  Culturas: {sorted(df['cultura'].unique())}")

    print("\nCarregando contorno do Brasil...")
    brazil = get_brazil_outline()

    culturas_principais = [
        "soja", "cana", "algodao", "cafe", "citrus", "arroz", "dende",
        "outras_lavouras_temporarias",
    ]

    print("\nGerando mapas individuais (safra e safrinha)...")
    for cultura in sorted(df["cultura"].unique()):
        for tipo in ["safra", "safrinha"]:
            plot_cultura_tipo(df, cultura, tipo, brazil)

    print("\nGerando painéis gerais de plantio...")
    for tipo in ["safra", "safrinha"]:
        plot_overview(df, brazil, culturas_principais, tipo)

    print(f"\nTodos os mapas salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
