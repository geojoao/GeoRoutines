"""
Avalia a acurácia do fenologia_brasil.parquet comparando SOS/EOS detectados
para a soja contra datas de plantio/colheita por estado (base Conab).

Saída: data/output/accuracy_conab.csv  +  data/output/accuracy_conab.png
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Tabela de referência Conab (DOY médio por estado — soja safra principal)
# Plantio em Out/Nov gera DOY > 274; colheita em Fev/Mar/Abr do ano seguinte
# representada como DOY < 150 (série continua após virada do ano)
# ---------------------------------------------------------------------------
CONAB = pd.DataFrame([
    # estado                  sos_ref  eos_ref   sos_label      eos_label
    ("Tocantins",               319,    60,  "Meio Nov",   "Início Mar"),
    ("Maranhão",                350,    85,  "Final Dez",  "Final Mar"),
    ("Piauí",                   335,    85,  "Final Nov",  "Final Mar"),
    ("Bahia",                   319,    85,  "Meio Nov",   "Final Mar"),
    ("Mato Grosso",             289,    45,  "Meio Out",   "Meio Fev"),
    ("Mato Grosso do Sul",      289,    60,  "Meio Out",   "Início Mar"),
    ("Goiás",                   319,    60,  "Meio Nov",   "Início Mar"),
    ("Minas Gerais",            319,    60,  "Meio Nov",   "Início Mar"),
    ("São Paulo",               289,    60,  "Meio Out",   "Início Mar"),
    ("Paraná",                  289,    60,  "Meio Out",   "Início Mar"),
    ("Santa Catarina",          319,   105,  "Meio Nov",   "Meio Abr"),
    ("Rio Grande do Sul",       335,   105,  "Final Nov",  "Meio Abr"),
], columns=["estado", "sos_ref", "eos_ref", "sos_label_ref", "eos_label_ref"])


def doy_diff_circular(a: float, b: float) -> float:
    """Diferença circular em dias (a - b), resultado em [-182, +183]."""
    d = (a - b) % 365
    if d > 182:
        d -= 365
    return d


# Centroides aproximados de cada estado com referência Conab (lon, lat)
_STATE_CENTROIDS = {
    "Tocantins":          (-48.3, -10.2),
    "Maranhão":           (-45.3,  -5.4),
    "Piauí":              (-43.0,  -7.0),
    "Bahia":              (-41.7, -12.9),
    "Mato Grosso":        (-55.9, -12.6),
    "Mato Grosso do Sul": (-54.5, -20.5),
    "Goiás":              (-49.3, -15.9),
    "Minas Gerais":       (-45.4, -18.1),
    "São Paulo":          (-48.7, -22.1),
    "Paraná":             (-51.6, -24.6),
    "Santa Catarina":     (-50.7, -27.4),
    "Rio Grande do Sul":  (-53.1, -30.0),
}


def assign_states(df_soja: pd.DataFrame, _states=None) -> pd.DataFrame:
    """Atribui estado a cada hexágono pelo centroide mais próximo (nearest-centroid)."""
    import h3

    estado_names = list(_STATE_CENTROIDS.keys())
    c_lon = np.array([v[0] for v in _STATE_CENTROIDS.values()])
    c_lat = np.array([v[1] for v in _STATE_CENTROIDS.values()])

    estados_assigned = []
    for hid in df_soja["id_hexagono"]:
        lat, lon = h3.cell_to_latlng(hid)
        dists = (lon - c_lon) ** 2 + (lat - c_lat) ** 2
        estados_assigned.append(estado_names[int(np.argmin(dists))])

    result = df_soja.copy()
    result["estado"] = estados_assigned
    return result


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula erro médio por estado vs referência Conab."""
    soja_safra = df[(df["cultura"] == "soja") & (df["tipo_safra"] == "safra")].copy()
    if soja_safra.empty:
        raise ValueError("Nenhum registro de soja/safra encontrado.")

    print(f"  {len(soja_safra):,} hex com soja/safra")

    print("Atribuindo estados por nearest-centroid...")
    soja_safra = assign_states(soja_safra)
    soja_safra = soja_safra.dropna(subset=["estado"])

    rows = []
    for _, ref in CONAB.iterrows():
        sub = soja_safra[soja_safra["estado"] == ref["estado"]]
        n = len(sub)
        if n == 0:
            rows.append({**ref.to_dict(), "n_hex": 0,
                         "sos_det_median": np.nan, "eos_det_median": np.nan,
                         "sos_erro_dias": np.nan, "eos_erro_dias": np.nan})
            continue

        sos_med = float(np.median(sub["sos_doy"]))
        eos_med = float(np.median(sub["eos_doy"]))
        sos_err = doy_diff_circular(sos_med, ref["sos_ref"])
        eos_err = doy_diff_circular(eos_med, ref["eos_ref"])
        rows.append({
            **ref.to_dict(),
            "n_hex": n,
            "sos_det_median": round(sos_med, 1),
            "eos_det_median": round(eos_med, 1),
            "sos_erro_dias": round(sos_err, 1),
            "eos_erro_dias": round(eos_err, 1),
        })

    return pd.DataFrame(rows)


def plot_accuracy(result: pd.DataFrame, output_path: Path):
    valid = result.dropna(subset=["sos_erro_dias", "eos_erro_dias"])
    if valid.empty:
        print("Sem dados suficientes para o gráfico.")
        return

    estados = valid["estado"].tolist()
    x = np.arange(len(estados))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Acurácia Fenologia vs Conab — Soja Safra Principal", fontsize=13, fontweight="bold")

    for ax, col, label, color in [
        (axes[0], "sos_erro_dias", "Plantio (SOS) — erro em dias", "#2196F3"),
        (axes[1], "eos_erro_dias", "Colheita (EOS) — erro em dias", "#FF5722"),
    ]:
        errs = valid[col].values
        colors = [color if abs(e) <= 16 else ("#FFC107" if abs(e) <= 32 else "#F44336") for e in errs]
        bars = ax.bar(x, errs, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(16, color="#FFC107", linewidth=0.6, linestyle="--", alpha=0.7)
        ax.axhline(-16, color="#FFC107", linewidth=0.6, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(estados, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Erro (dias detectado − referência)")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-60, 60)
        # Adiciona n_hex
        for xi, (e, n) in enumerate(zip(errs, valid["n_hex"].values)):
            ax.text(xi, e + (2 if e >= 0 else -5), f"n={n}", ha="center", va="bottom", fontsize=6)

    # Legenda
    patch_ok = mpatches.Patch(color="#2196F3", label="≤ 16 dias (≈ 1 composição)")
    patch_warn = mpatches.Patch(color="#FFC107", label="16–32 dias")
    patch_err = mpatches.Patch(color="#F44336", label="> 32 dias")
    axes[0].legend(handles=[patch_ok, patch_warn, patch_err], fontsize=8)

    # Métricas globais
    mae_sos = np.nanmean(np.abs(valid["sos_erro_dias"]))
    mae_eos = np.nanmean(np.abs(valid["eos_erro_dias"]))
    fig.text(0.5, -0.02,
             f"MAE Plantio (SOS): {mae_sos:.1f} dias  |  MAE Colheita (EOS): {mae_eos:.1f} dias",
             ha="center", fontsize=10, color="#333333", fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Gráfico salvo: {output_path}")


def main():
    input_path = Path("data/output/fenologia_brasil.parquet")
    if not input_path.exists():
        print("fenologia_brasil.parquet não encontrado. Rode extract_phenology.py primeiro.")
        sys.exit(1)

    print("Carregando fenologia_brasil.parquet...")
    df = pd.read_parquet(input_path)
    print(f"  {len(df):,} registros")

    print("\nAvaliando acurácia vs Conab...")
    result = evaluate(df)

    out_csv = Path("data/output/accuracy_conab.csv")
    result.to_csv(out_csv, index=False)
    print(f"\nResultado salvo em {out_csv}")
    print(result[["estado", "n_hex", "sos_ref", "sos_det_median", "sos_erro_dias",
                  "eos_ref", "eos_det_median", "eos_erro_dias"]].to_string(index=False))

    mae_sos = result["sos_erro_dias"].abs().mean()
    mae_eos = result["eos_erro_dias"].abs().mean()
    print(f"\nMAE Plantio (SOS): {mae_sos:.1f} dias")
    print(f"MAE Colheita (EOS): {mae_eos:.1f} dias")

    plot_accuracy(result, Path("data/output/accuracy_conab.png"))


if __name__ == "__main__":
    main()
