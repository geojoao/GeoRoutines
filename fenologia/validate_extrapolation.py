"""
Validação visual do método de extrapolação por shape-matching (k-NN).

Para cada hexágono amostrado:
  - Detecta ciclos com extract_phenometrics
  - Roda validate_extrapolation (leave-one-out, τ ∈ {0.25, 0.50, 0.75})
  - Gera um painel por ciclo mostrando previsão vs real
  - Gera um painel de resumo com MAE e erro de EOS

Uso:
    uv run python validate_extrapolation.py soja 5 42 valida
    uv run python validate_extrapolation.py soja 5 42 valida --hexagonos 858b8427fffffff,85...
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
PARTS = Path("data/output/_parts")
OUT_DIR = Path("data/output/diag_extrap")

COLORS_TAU = {0.25: "#E74C3C", 0.50: "#F39C12", 0.75: "#27AE60"}


def load_cultura(cultura):
    col = f"evi_medio_{cultura}"
    dfs = []
    for p in sorted(PARTS.glob("*.parquet")):
        d = pd.read_parquet(p, columns=["id_hexagono", "data", col])
        d = d.dropna(subset=[col])
        if len(d):
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(["id_hexagono", "data"])
    return df, col


def run_hex(hex_id, grp, col, cultura, iqr_factor=1.5):
    grp = grp.sort_values("data")
    ts = grp.rename(columns={"data": "datetime", col: "NDVI_mean"})[["datetime", "NDVI_mean"]]
    min_days = MIN_CYCLE_DAYS.get(cultura, 90)

    res = extract_phenometrics(
        ts, ndvi_column="NDVI_mean",
        min_cycle_length_days=min_days,
        smoothing_method="both",
        quality_threshold=0.60,
    )
    if not res["success"]:
        return None, None

    ndvi_arr  = ts["NDVI_mean"].values.astype(float)
    dates_arr = ts["datetime"].values
    smooth    = adaptive_smoothing(ndvi_arr, dates_arr, method="both")

    val    = validate_extrapolation(res["cycles"], iqr_factor=iqr_factor)
    extrap = extrapolate_terminal_cycle(smooth, dates_arr, res["cycles"], iqr_factor=iqr_factor)

    return res, val, smooth, ndvi_arr, dates_arr, extrap


# ─── plots ────────────────────────────────────────────────────────────────────

def plot_validation(hex_id, res, val, smooth, ndvi_arr, dates_arr, extrap, cultura, out_path):
    if not val["success"] or not val["results"]:
        return

    n_cycles = len(val["results"])
    n_rows   = n_cycles + 1  # +1 para o painel de extrap / série completa
    fig = plt.figure(figsize=(18, 3.5 * n_rows), constrained_layout=True)
    gs  = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.05)

    # ── Painel 0: série completa + ciclos + extrap ─────────────────────────
    ax0 = fig.add_subplot(gs[0])
    dates_dt = pd.to_datetime(dates_arr)
    ax0.plot(dates_dt, ndvi_arr, color="0.7", lw=0.8, alpha=0.6, label="EVI bruto")
    ax0.plot(dates_dt, smooth,   color="#1f77b4", lw=1.8, label="EVI suave")

    # Ciclos completos aceitos
    pal = plt.cm.tab10(np.linspace(0, 1, 10))
    for ci, c in enumerate(res["cycles"]):
        if not c.get("fit_success"):
            continue
        cp  = c["curve_params"]
        dur = c["cycle_length_days"]
        cs  = pd.Timestamp(c["cycle_start"])
        xs  = np.linspace(0, dur, 200)
        ys  = double_logistic(xs, cp["amplitude"], cp["m1"], cp["k1"],
                              cp["m2"], cp["k2"], cp["offset"])
        xdt = [cs + pd.Timedelta(days=float(x)) for x in xs]
        color = pal[ci % 10]
        ls    = "-" if not c.get("at_series_end") else "--"
        ax0.plot(xdt, ys, ls, color=color, lw=2, alpha=0.85)

    # Extrapolação terminal
    if extrap and extrap.get("success"):
        c_info = extrap["curve"]
        cdates = pd.to_datetime(c_info["dates"])
        cvals  = np.array(c_info["values"])
        cstd   = np.array(c_info["std"])
        is_fc  = np.array(c_info["is_forecast"])

        ax0.plot(cdates[~is_fc], cvals[~is_fc], color="orange", lw=2.5, alpha=0.9, label="Terminal observado")
        ax0.plot(cdates[is_fc],  cvals[is_fc],  color="orange", lw=2.5, ls="--",
                 label=f"Extrap. n={extrap['n_cycles_used']} (-{extrap['n_outliers_removed']} outliers)")
        ax0.fill_between(cdates[is_fc],
                         cvals[is_fc] - cstd[is_fc],
                         cvals[is_fc] + cstd[is_fc],
                         color="orange", alpha=0.15)
        ax0.axvline(extrap["series_end_date"],   color="gray",   ls=":",  lw=1.5, label="Fim da série")
        ax0.axvline(extrap["forecast_eos_date"], color="orange", ls=":",  lw=1.5,
                    label=f"EOS previsto ({str(extrap['forecast_eos_date'])[:10]})")
        tau   = extrap["tau_obs"]
        stype = extrap.get("season_type_filter") or "todos"
        ax0.set_title(f"{hex_id} · {cultura} · τ_obs={tau:.0%} · "
                      f"season_type={stype} · incerteza={extrap['uncertainty_days']:.0f}d",
                      fontsize=10, fontweight="bold")
    else:
        reason = extrap.get("reason", "?") if extrap else "N/A"
        ax0.set_title(f"{hex_id} · {cultura} · sem extrap ({reason})", fontsize=10)

    ax0.set_ylim(0, 1.0)
    ax0.legend(fontsize=7, loc="upper right", ncol=3)
    ax0.grid(alpha=0.2)
    ax0.set_ylabel("EVI")

    # ── Painéis 1..n: validação leave-one-out por ciclo ───────────────────
    n_pts = 100
    for row_i, cyc_res in enumerate(val["results"]):
        ax = fig.add_subplot(gs[row_i + 1])

        y_full = cyc_res["y_full_norm"]
        t_full = np.linspace(0, 1, n_pts)
        dur    = cyc_res["dur_days"]

        ax.plot(t_full, y_full, "k-", lw=2, alpha=0.7, label=f"Ciclo {cyc_res['cycle_num']} real")

        for tr in cyc_res["truncations"]:
            tau   = tr["tau"]
            m_obs = tr["m_obs"]
            col_t = COLORS_TAU[tau]

            # Prefixo observado
            ax.plot(t_full[:m_obs], y_full[:m_obs], color=col_t, lw=3, alpha=0.5)
            # Cauda prevista
            t_tail = np.linspace(m_obs / n_pts, 1, len(tr["pred_tail"]))
            n_info = f"n={tr.get('n_used','?')} (-{tr.get('n_outliers_removed', 0)})"
            ax.plot(t_tail, tr["pred_tail"], color=col_t, ls="--", lw=2,
                    label=f"τ={tau:.0%}  MAE={tr['mae']:.3f}  ΔEOS={tr['eos_error_days'] or '?'}d  {n_info}")
            ax.fill_between(t_tail,
                            tr["pred_tail"] - tr["pred_std"],
                            tr["pred_tail"] + tr["pred_std"],
                            color=col_t, alpha=0.10)
            # Linha de truncação
            ax.axvline(m_obs / n_pts, color=col_t, ls=":", lw=1, alpha=0.6)

        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel("τ (fração normalizada do ciclo)")
        ax.set_ylabel("EVI norm.")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        title_parts = [f"ciclo {cyc_res['cycle_num']}",
                       f"{cyc_res['season_type']}",
                       f"{cyc_res['dur_days']:.0f} dias"]
        ax.set_title("  ·  ".join(title_parts), fontsize=9)

    fig.suptitle(f"Validação extrapolação — {hex_id} / {cultura}", fontsize=12, y=1.002)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path}")


def main():
    cultura    = sys.argv[1] if len(sys.argv) > 1 else "soja"
    n_hex      = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed       = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    prefix     = sys.argv[4] if len(sys.argv) > 4 else "valida"
    iqr_factor = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5

    # suporte a --hexagonos hex1,hex2,...
    forced_hexes = []
    for a in sys.argv[1:]:
        if a.startswith("--hexagonos="):
            forced_hexes = a.split("=", 1)[1].split(",")
        elif a == "--hexagonos" and sys.argv.index(a) + 1 < len(sys.argv):
            forced_hexes = sys.argv[sys.argv.index(a) + 1].split(",")

    print(f"Carregando {cultura}...", flush=True)
    df, col = load_cultura(cultura)

    counts = df.groupby("id_hexagono").size()
    good   = counts[counts >= 60].index.tolist()

    if forced_hexes:
        sample = [h for h in forced_hexes if h in good]
    else:
        rng    = np.random.default_rng(seed)
        sample = list(rng.choice(good, size=min(n_hex, len(good)), replace=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for hex_id in sample:
        print(f"\n{hex_id}", flush=True)
        grp = df[df["id_hexagono"] == hex_id]
        out = run_hex(hex_id, grp, col, cultura, iqr_factor=iqr_factor)
        if out[0] is None:
            print("  sem picos detectados")
            continue
        res, val, smooth, ndvi_arr, dates_arr, extrap = out
        n_complete = sum(1 for c in res["cycles"]
                         if c.get("fit_success") and not c.get("at_series_end") and not c.get("at_series_start"))
        print(f"  {len(res['cycles'])} ciclos, {n_complete} completos")

        if not val["success"]:
            print(f"  validação: {val['reason']}")
        else:
            for cr in val["results"]:
                for tr in cr["truncations"]:
                    eos_e = tr["eos_error_days"]
                    n_u = tr.get("n_used", "?")
                    n_r = tr.get("n_outliers_removed", 0)
                    print(f"    ciclo {cr['cycle_num']} τ={tr['tau']:.0%}: "
                          f"MAE={tr['mae']:.3f}  ΔEOS={eos_e}d  [n={n_u} -out={n_r}]")

        out_path = OUT_DIR / f"{prefix}_{cultura}_{hex_id}.png"
        plot_validation(hex_id, res, val, smooth, ndvi_arr, dates_arr, extrap, cultura, out_path)

    print(f"\nPlots em {OUT_DIR}")


if __name__ == "__main__":
    main()
