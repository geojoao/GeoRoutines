"""
Diagnóstico visual do phenophase: amostra hexágonos, roda extract_phenometrics
com os MESMOS parâmetros do extract_phenology.py / dashboard, e plota
EVI bruto + suavizado + troughs + gaussianas por ciclo + SOS/POS/EOS.

Uso: uv run python diag_phenophase.py <cultura> <n_hex> <seed> <out_prefix>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from fenologia.phenophase import extract_phenometrics, adaptive_smoothing, double_logistic

MIN_CYCLE_DAYS = {
    "soja": 120, "cana": 150, "arroz": 100, "algodao": 110, "cafe": 120,
    "citrus": 120, "dende": 120, "outras_lavouras_temporarias": 100,
    "outras_lavouras_perenes": 120, "segunda_safra": 90,
    "segunda_safra_algodao": 100, "segunda_safra_outras_temporarias": 90,
}
MAX_CYCLE_DAYS = {
    "soja": 180, "cana": 420, "arroz": 160, "algodao": 220, "cafe": 400,
    "citrus": 400, "dende": 400, "outras_lavouras_temporarias": 200,
    "outras_lavouras_perenes": 400, "segunda_safra": 150,
    "segunda_safra_algodao": 220, "segunda_safra_outras_temporarias": 180,
}
MIN_EVI_AMPLITUDE = 0.08
MIN_R2 = 0.80
MIN_GROWING_DAYS = 35  # largura mínima do pico (SOS->EOS) — rejeita spikes degenerados

PARTS = Path("data/output/_parts")


def load_culture(cultura):
    col = f"evi_medio_{cultura}"
    dfs = []
    for p in sorted(PARTS.glob("*.parquet")):
        d = pd.read_parquet(p, columns=["id_hexagono", "data", col])
        d = d.dropna(subset=[col])
        if len(d):
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(["id_hexagono", "data"])
    return df, col


def plot_hex(ax, hex_id, ts_df, cultura):
    min_days = MIN_CYCLE_DAYS.get(cultura, 90)
    max_days = MAX_CYCLE_DAYS.get(cultura, 400)
    res = extract_phenometrics(ts_df, ndvi_column="NDVI_mean",
                               min_cycle_length_days=min_days,
                               smoothing_method="both", quality_threshold=0.85)
    ndvi = ts_df["NDVI_mean"].values.astype(float)
    dates = ts_df["datetime"].values
    smooth = adaptive_smoothing(ndvi, dates, method="both")
    dt = pd.to_datetime(dates)

    ax.plot(dt, ndvi, color="0.6", lw=0.8, alpha=0.6, label="EVI bruto")
    ax.plot(dt, smooth, color="#1f77b4", lw=1.8, label="EVI suave")

    peaks = res["diagnostics"].get("peak_indices", res["diagnostics"].get("troughs_indices", []))
    if peaks:
        ax.scatter(dt[peaks], ndvi[peaks], marker="^", color="green", s=40, zorder=5, label="picos")

    n_ok = 0
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in res.get("cycles", []):
        if not c.get("fit_success"):
            continue
        L = c["cycle_length_days"]
        amp = c["gaussian_params"]["amplitude"]
        r2 = c["r_squared"]
        pdys = c["phenophase_days"]
        growing = pdys["eos_days"] - pdys["sos_days"]
        kept = (MIN_GROWING_DAYS <= growing <= max_days) and amp >= MIN_EVI_AMPLITUDE and r2 >= MIN_R2
        color = palette[n_ok % 10] if kept else "0.5"
        cs = pd.Timestamp(c["cycle_start"]); ce = pd.Timestamp(c["cycle_end"])
        gp = c["gaussian_params"]
        xs = np.linspace(0, L, 80)
        ys = double_logistic(xs, gp["amplitude"], gp["m1"], gp["k1"], gp["m2"], gp["k2"], gp["offset"])
        gdt = [cs + pd.Timedelta(days=float(x)) for x in xs]
        style = "-" if kept else ":"
        ax.plot(gdt, ys, style, color=color, lw=2 if kept else 1, alpha=0.9 if kept else 0.5)
        ph = c["phenophase_dates"]; pv = c["phenophase_values"]
        if kept:
            ax.axvline(pd.Timestamp(ph["sos"]), color="green", ls="--", lw=1, alpha=0.7)
            ax.axvline(pd.Timestamp(ph["eos"]), color="red", ls="--", lw=1, alpha=0.7)
            ax.scatter([pd.Timestamp(ph["pos"])], [pv["pos_ndvi"]], marker="*", s=180,
                       color=color, edgecolors="k", zorder=6)
            ax.text(pd.Timestamp(ph["pos"]), pv["pos_ndvi"] + 0.04,
                    f'R²={r2:.2f}\n{c["season_type"][:4]}\n{int(L)}d',
                    fontsize=6, ha="center", color="k")
            n_ok += 1

    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2)
    ax.set_title(f"{hex_id} · {cultura} · {n_ok} ciclos válidos · {len(troughs)} vales · {len(ndvi)} pts",
                 fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    return n_ok


def main():
    cultura = sys.argv[1] if len(sys.argv) > 1 else "soja"
    n_hex = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    prefix = sys.argv[4] if len(sys.argv) > 4 else "diag"

    df, col = load_culture(cultura)
    counts = df.groupby("id_hexagono").size()
    # Só hexágonos com cobertura decente (>= 60 pontos ao longo de 5 anos)
    good = counts[counts >= 60].index.tolist()
    rng = np.random.default_rng(seed)
    sample = rng.choice(good, size=min(n_hex, len(good)), replace=False)

    per_fig = 3
    out_dir = Path("data/output/diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fig_i in range(0, len(sample), per_fig):
        chunk = sample[fig_i:fig_i + per_fig]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(16, 3.4 * len(chunk)))
        if len(chunk) == 1:
            axes = [axes]
        for ax, hex_id in zip(axes, chunk):
            sub = df[df["id_hexagono"] == hex_id].sort_values("data")
            ts = sub.rename(columns={"data": "datetime", col: "NDVI_mean"})[["datetime", "NDVI_mean"]]
            plot_hex(ax, hex_id, ts, cultura)
        fig.tight_layout()
        out = out_dir / f"{prefix}_{cultura}_{fig_i // per_fig}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out))
        print(f"Salvo: {out}", flush=True)
    print("HEXES:", ",".join(sample))


if __name__ == "__main__":
    main()
