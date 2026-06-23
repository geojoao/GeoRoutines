"""
Validação visual da extrapolação de pico terminal — método de prior de descida.

Para cada hexágono amostrado:
  - Detecta ciclos com extract_phenometrics
  - Painel 0: série EVI + logísticas ajustadas + curva extrapolada do terminal
  - Painéis 1..n: validação leave-one-out — curva true vs. curva reconstruída
                  com prior; marca EOS verdadeiro e previsto

Uso:
    cd fenologia
    uv run python validate_extrapolation.py soja 5 42 v3
    uv run python validate_extrapolation.py soja 5 42 v3 --hexagonos 85a8ed9bfffffff,858b8217fffffff
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from fenologia.phenophase import (
    extract_phenometrics, adaptive_smoothing,
    extrapolate_terminal_cycle, validate_extrapolation,
    double_logistic,
)

MIN_CYCLE_DAYS = {
    "soja": 120, "cana": 150, "arroz": 100, "algodao": 110,
    "cafe": 120, "citrus": 120, "dende": 120,
    "outras_lavouras_temporarias": 100, "outras_lavouras_perenes": 120,
    "segunda_safra": 90, "segunda_safra_algodao": 100,
    "segunda_safra_outras_temporarias": 90,
}
PARTS   = Path("data/output/_parts")
OUT_DIR = Path("data/output/diag_extrap")

COLORS_DELTA = {10: "#E74C3C", 30: "#27AE60"}


def load_cultura(cultura):
    col = f"evi_medio_{cultura}"
    dfs = []
    for p in sorted(PARTS.glob("*.parquet")):
        d = pd.read_parquet(p, columns=["id_hexagono", "data", col]).dropna(subset=[col])
        if len(d):
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(["id_hexagono", "data"])
    return df, col


def run_hex(hex_id, grp, col, cultura):
    grp = grp.sort_values("data")
    ts  = grp.rename(columns={"data": "datetime", col: "NDVI_mean"})[["datetime", "NDVI_mean"]]
    min_days = MIN_CYCLE_DAYS.get(cultura, 90)

    res = extract_phenometrics(
        ts, ndvi_column="NDVI_mean",
        min_cycle_length_days=min_days,
        smoothing_method="both",
        quality_threshold=0.60,
    )
    if not res["success"]:
        return None

    ndvi_arr  = ts["NDVI_mean"].values.astype(float)
    dates_arr = ts["datetime"].values
    smooth    = adaptive_smoothing(ndvi_arr, dates_arr, method="both")

    val    = validate_extrapolation(res["cycles"])
    extrap = extrapolate_terminal_cycle(smooth, dates_arr, res["cycles"])

    return res, val, smooth, ndvi_arr, dates_arr, extrap


def plot_hex(hex_id, res, val, smooth, ndvi_arr, dates_arr, extrap, cultura, out_path):
    n_val_cycles = len(val["results"]) if val["success"] else 0
    n_rows = 1 + n_val_cycles
    fig = plt.figure(figsize=(18, 3.5 * n_rows), constrained_layout=True)
    gs  = gridspec.GridSpec(n_rows, 1, figure=fig)

    # ── Painel 0: série completa + extrap ─────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    dates_dt = pd.to_datetime(dates_arr)
    ax0.plot(dates_dt, ndvi_arr, color="0.75", lw=0.8, alpha=0.6, label="EVI bruto")
    ax0.plot(dates_dt, smooth,   color="#1f77b4", lw=1.5, label="EVI suave")

    pal = plt.cm.tab10(np.linspace(0, 1, 10))
    for ci, c in enumerate(res["cycles"]):
        if not c.get("fit_success"):
            continue
        cp  = c["curve_params"]
        dur = c["cycle_length_days"]
        cs  = pd.Timestamp(c["cycle_start"])
        t   = np.linspace(0, dur, 300)
        y   = double_logistic(t, cp["amplitude"], cp["m1"], cp["k1"],
                              cp["m2"], cp["k2"], cp["offset"])
        xdt = [cs + pd.Timedelta(days=float(x)) for x in t]
        ls  = "--" if c.get("at_series_end") or c.get("at_series_start") else "-"
        ax0.plot(xdt, y, ls, color=pal[ci % 10], lw=1.8, alpha=0.85)

    if extrap and extrap.get("success"):
        cv      = extrap["curve"]
        cdates  = pd.to_datetime(cv["dates"])
        cvals   = np.array(cv["values"])
        cup     = np.array(cv["values_upper"])
        clo     = np.array(cv["values_lower"])
        is_fc   = np.array(cv["is_forecast"])

        ax0.plot(cdates[is_fc], cvals[is_fc], color="darkorange", lw=2.5,
                 ls="--", label="Extrap. prior descida")
        ax0.fill_between(cdates[is_fc], clo[is_fc], cup[is_fc],
                         color="darkorange", alpha=0.18, label="±1 std prior")
        ax0.axvline(extrap["series_end_date"],   color="gray",       ls=":", lw=1.5)
        ax0.axvline(extrap["forecast_eos_date"], color="darkorange", ls=":", lw=1.5,
                    label=f"EOS prev. {str(extrap['forecast_eos_date'])[:10]}")

        pr = extrap["prior"]
        title = (f"{hex_id} · {cultura} · n_prior={extrap['n_prior_cycles']} "
                 f"({extrap['season_type_filter'] or 'todos'}) · "
                 f"peak_obs={extrap['peak_was_observed']} · "
                 f"k2={pr['k2_median']:.3f}±{pr['k2_std']:.3f} · "
                 f"half_r={pr['half_right_median']:.0f}±{pr['half_right_std']:.0f}d")
    else:
        reason = extrap.get("reason", "N/A") if extrap else "N/A"
        title  = f"{hex_id} · {cultura} · sem extrap terminal ({reason})"

    ax0.set_title(title, fontsize=9, fontweight="bold")
    ax0.set_ylim(0, 1.0)
    ax0.legend(fontsize=7, loc="upper right", ncol=3)
    ax0.grid(alpha=0.2)
    ax0.set_ylabel("EVI")

    # ── Painéis leave-one-out ─────────────────────────────────────────────
    if not val["success"]:
        fig.suptitle(f"Validação — {hex_id} / {cultura}", fontsize=12)
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return

    for row_i, cr in enumerate(val["results"]):
        ax = fig.add_subplot(gs[row_i + 1])

        t = cr["t_dense"]
        t_rel = t / cr["dur_days"]  # normalizado 0-1 para comparação
        pos_rel  = next((cr["prior_half_r"] + min(tr["obs_days"] for tr in cr["truncations"]))
                        / cr["dur_days"] for _ in [1])   # rough line
        eos_true = cr["eos_true_days"] / cr["dur_days"]

        ax.plot(t_rel, cr["y_true"],    "k-",  lw=2,   alpha=0.7, label=f"Real (ciclo {cr['cycle_num']})")
        ax.plot(t_rel, cr["y_central"], color="darkorange", lw=2, ls="--", label="Prior central")
        ax.fill_between(t_rel, cr["y_lower"], cr["y_upper"],
                        color="darkorange", alpha=0.15, label=f"±1std (hr={cr['prior_half_r']:.0f}±{cr['prior_half_r_std']:.0f}d)")

        # Marca EOS verdadeiro e previsto por truncação
        ax.axvline(eos_true, color="black", ls=":", lw=1.5, label=f"EOS real")
        for tr in cr["truncations"]:
            delta  = int(tr["trunc_label"].replace("POS+","").replace("d",""))
            col_t  = COLORS_DELTA.get(delta, "gray")
            obs_r  = tr["obs_days"] / cr["dur_days"]
            eos_r  = tr["eos_pred_days"] / cr["dur_days"]
            err    = tr["eos_error_days"]
            ax.axvline(obs_r, color=col_t, ls="-",  lw=1,   alpha=0.5)
            ax.axvline(eos_r, color=col_t, ls="--", lw=1.5,
                       label=f"{tr['trunc_label']}  ΔEOS={err:+.0f}d")

        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("τ (fração normalizada do ciclo)")
        ax.set_ylabel("EVI")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        ax.set_title(f"ciclo {cr['cycle_num']} · {cr['season_type']} · "
                     f"{cr['dur_days']:.0f}d · n_bank={cr['n_bank']}",
                     fontsize=9)

    fig.suptitle(f"Validação prior descida — {hex_id} / {cultura}", fontsize=12, y=1.002)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path}")


def main():
    cultura = sys.argv[1] if len(sys.argv) > 1 else "soja"
    n_hex   = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed    = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    prefix  = sys.argv[4] if len(sys.argv) > 4 else "v3"

    forced_hexes = []
    for a in sys.argv[1:]:
        if a.startswith("--hexagonos="):
            forced_hexes = a.split("=", 1)[1].split(",")
        elif a == "--hexagonos" and sys.argv.index(a) + 1 < len(sys.argv):
            forced_hexes = sys.argv[sys.argv.index(a) + 1].split(",")

    print(f"Carregando {cultura}...", flush=True)
    df, col = load_cultura(cultura)
    counts  = df.groupby("id_hexagono").size()
    good    = counts[counts >= 60].index.tolist()

    if forced_hexes:
        sample = [h for h in forced_hexes if h in good]
    else:
        rng    = np.random.default_rng(seed)
        sample = list(rng.choice(good, size=min(n_hex, len(good)), replace=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for hex_id in sample:
        print(f"\n{hex_id}", flush=True)
        grp = df[df["id_hexagono"] == hex_id]
        out = run_hex(hex_id, grp, col, cultura)
        if out is None:
            print("  sem picos detectados")
            continue

        res, val, smooth, ndvi_arr, dates_arr, extrap = out
        n_complete = sum(1 for c in res["cycles"]
                         if c.get("fit_success")
                         and not c.get("at_series_end")
                         and not c.get("at_series_start"))
        term_str = "SIM" if extrap and extrap.get("success") else "NÃO"
        print(f"  {len(res['cycles'])} ciclos, {n_complete} completos, terminal={term_str}")

        if extrap and extrap.get("success"):
            days_ahead = (pd.Timestamp(extrap["forecast_eos_date"])
                          - pd.Timestamp(extrap["series_end_date"])).days
            print(f"  EOS previsto em {days_ahead}d · incerteza={extrap['uncertainty_days']}d")

        if val["success"]:
            print(f"  Validação ({val['n_cycles']} ciclos):")
            for cr in val["results"]:
                for tr in cr["truncations"]:
                    print(f"    ciclo {cr['cycle_num']:2d} {tr['trunc_label']:8s}  "
                          f"ΔEOS={tr['eos_error_days']:+6.1f}d")

        out_path = OUT_DIR / f"{prefix}_{cultura}_{hex_id}.png"
        plot_hex(hex_id, res, val, smooth, ndvi_arr, dates_arr, extrap, cultura, out_path)

    print(f"\nPlots em {OUT_DIR}")


if __name__ == "__main__":
    main()
