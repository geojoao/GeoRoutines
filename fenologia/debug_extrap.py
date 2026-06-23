"""
Debug profundo da extrapolação de pico terminal.

Investiga hipóteses sobre por que a extrapolação parece "defasada"/errada:

H1. Inconsistência de normalização:
    - validate_extrapolation normaliza o prefixo da query pela amplitude do
      ciclo COMPLETO (conhece o pico futuro -> "cola")
    - extrapolate_terminal_cycle normaliza pela amplitude OBSERVADA (não vê o
      pico futuro). Os dois medem coisas diferentes.

H2. Quando o pico ainda não foi observado, obs_max != pico verdadeiro, então
    obs_norm força o último ponto a ~1.0 -> shape matching comparando maçã/laranja.

H3. tau_obs é estimado por (dur_obs / mediana_dur). Se errado, remaining_days
    estica/encolhe a cauda -> EOS deslocado no tempo.

Uso:
    uv run python debug_extrap.py soja 85a8de93fffffff
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from fenologia.phenophase import (
    extract_phenometrics, adaptive_smoothing,
    extrapolate_terminal_cycle, double_logistic, _cycle_normalized_curve,
)

MIN_CYCLE_DAYS = {"soja": 120}
PARTS = Path("data/output/_parts")


def load_cultura(cultura):
    col = f"evi_medio_{cultura}"
    dfs = []
    for p in sorted(PARTS.glob("*.parquet")):
        d = pd.read_parquet(p, columns=["id_hexagono", "data", col]).dropna(subset=[col])
        if len(d):
            dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(["id_hexagono", "data"])
    return df, col


def main():
    cultura = sys.argv[1] if len(sys.argv) > 1 else "soja"
    hex_id  = sys.argv[2] if len(sys.argv) > 2 else "85a8de93fffffff"

    df, col = load_cultura(cultura)
    grp = df[df["id_hexagono"] == hex_id].sort_values("data")
    ts = grp.rename(columns={"data": "datetime", col: "NDVI_mean"})[["datetime", "NDVI_mean"]]

    res = extract_phenometrics(
        ts, ndvi_column="NDVI_mean",
        min_cycle_length_days=MIN_CYCLE_DAYS.get(cultura, 90),
        smoothing_method="both", quality_threshold=0.60,
    )
    print(f"\n=== {hex_id} / {cultura} ===")
    print(f"success={res['success']}  n_cycles={len(res.get('cycles', []))}")

    cycles = res["cycles"]
    ndvi_arr  = ts["NDVI_mean"].values.astype(float)
    dates_arr = ts["datetime"].values
    smooth    = adaptive_smoothing(ndvi_arr, dates_arr, method="both")

    print("\n--- Inventário de ciclos ---")
    for c in cycles:
        if not c.get("fit_success"):
            print(f"  ciclo {c.get('cycle', {}).get('cycle_num','?')}: FIT FALHOU")
            continue
        flags = []
        if c.get("at_series_start"): flags.append("START")
        if c.get("at_series_end"):   flags.append("END")
        pd_ = c["phenophase_days"]
        print(f"  ciclo {c['cycle_num']:2d}  {c['season_type']:12s}  "
              f"dur={c['cycle_length_days']:5.0f}d  "
              f"SOS={pd_['sos_days']:.0f} POS={pd_['pos_days']:.0f} EOS={pd_['eos_days']:.0f}  "
              f"R2={c['r_squared']:.2f}  amp={c['curve_params']['amplitude']:.3f}  "
              f"{' '.join(flags)}")

    # ── Localiza terminal ──────────────────────────────────────────────────
    terminal = None
    for c in reversed(cycles):
        at_end = (c.get("at_series_end", False) if c.get("fit_success")
                  else c.get("cycle", {}).get("at_series_end", False))
        if at_end:
            terminal = c
            break

    if terminal is None:
        print("\n[!] Nenhum ciclo terminal (at_series_end). "
              "A extrapolação NÃO se aplica a este hexágono.")
        return

    print("\n--- CICLO TERMINAL ---")
    if terminal.get("fit_success"):
        t_start = np.datetime64(terminal["cycle_start"])
        start_idx = int(np.searchsorted(dates_arr, t_start))
        print(f"  cycle_num={terminal['cycle_num']}  season={terminal['season_type']}")
        print(f"  cycle_start={terminal['cycle_start']}  start_idx={start_idx}")
        pd_ = terminal["phenophase_days"]
        print(f"  fit POS_days={pd_['pos_days']:.0f}  EOS_days={pd_['eos_days']:.0f}  "
              f"dur_fit={terminal['cycle_length_days']:.0f}d")
    else:
        start_idx = terminal["cycle"]["start_idx"]
        print(f"  (fit falhou) start_idx={start_idx}")

    obs_evi   = smooth[start_idx:].astype(float)
    obs_dates = dates_arr[start_idx:]
    obs_dur   = float((obs_dates[-1] - obs_dates[0]) / np.timedelta64(1, "D"))
    obs_min, obs_max = float(obs_evi.min()), float(obs_evi.max())
    argmax_pos = int(np.argmax(obs_evi))

    print(f"\n  observado: {len(obs_evi)} pts, "
          f"{str(obs_dates[0])[:10]} -> {str(obs_dates[-1])[:10]} ({obs_dur:.0f}d)")
    print(f"  obs_min={obs_min:.3f}  obs_max={obs_max:.3f}  span={obs_max-obs_min:.3f}")
    print(f"  argmax em idx {argmax_pos}/{len(obs_evi)-1} "
          f"({'PICO JÁ PASSOU' if argmax_pos < len(obs_evi)-1 else 'AINDA SUBINDO -> pico não observado'})")
    print(f"  último valor observado = {obs_evi[-1]:.3f}  "
          f"(normalizado seria {(obs_evi[-1]-obs_min)/(obs_max-obs_min+1e-9):.2f})")

    # ── H2/H3: estimativa de tau_obs ───────────────────────────────────────
    db = [c for c in cycles if c.get("fit_success")
          and not c.get("at_series_end") and not c.get("at_series_start")]
    med_dur = float(np.median([c["cycle_length_days"] for c in db]))
    tau_obs = float(np.clip(obs_dur / med_dur, 0.05, 0.95))
    print(f"\n  [H3] mediana_dur_banco={med_dur:.0f}d  ->  "
          f"tau_obs estimado = {obs_dur:.0f}/{med_dur:.0f} = {tau_obs:.2f}")
    # qual o tau "verdadeiro" segundo o fit do próprio terminal?
    if terminal.get("fit_success"):
        true_dur = terminal["cycle_length_days"]
        print(f"       dur_fit do terminal={true_dur:.0f}d  ->  "
              f"tau_verdadeiro ~ {obs_dur/true_dur:.2f}")

    # ── H1: o que a validação faz vs produção ──────────────────────────────
    print("\n--- [H1] Normalização: validação vs produção ---")
    print("  validação:  prefixo normalizado pela amplitude do ciclo COMPLETO (vê o pico)")
    print("  produção:   prefixo normalizado por obs_min/obs_max (NÃO vê o pico futuro)")
    if terminal.get("fit_success"):
        y_full_norm = _cycle_normalized_curve(terminal)
        if y_full_norm is not None:
            m_obs = max(3, min(98, int(round(tau_obs * 100))))
            # validação-style: prefixo do ciclo completo normalizado
            valid_prefix = y_full_norm[:m_obs]
            # produção-style: observado normalizado por obs
            obs_norm = (obs_evi - obs_min) / (obs_max - obs_min + 1e-9)
            t_src = np.linspace(0, 1, len(obs_norm))
            t_dst = np.linspace(0, 1, m_obs)
            prod_prefix = np.interp(t_dst, t_src, obs_norm)
            print(f"\n  m_obs={m_obs}")
            print(f"  valid_prefix [0], [meio], [-1] = "
                  f"{valid_prefix[0]:.2f}, {valid_prefix[m_obs//2]:.2f}, {valid_prefix[-1]:.2f}")
            print(f"  prod_prefix  [0], [meio], [-1] = "
                  f"{prod_prefix[0]:.2f}, {prod_prefix[m_obs//2]:.2f}, {prod_prefix[-1]:.2f}")
            diff = float(np.sqrt(np.mean((valid_prefix - prod_prefix) ** 2)))
            print(f"  >>> RMSE entre os dois prefixos = {diff:.3f}  "
                  f"(se alto, validação NÃO representa a produção)")

    # ── Resultado real da produção ─────────────────────────────────────────
    print("\n--- Resultado de extrapolate_terminal_cycle (produção) ---")
    extrap = extrapolate_terminal_cycle(smooth, dates_arr, cycles)
    if not extrap.get("success"):
        print(f"  FALHOU: {extrap.get('reason')}")
        return
    print(f"  tau_obs={extrap['tau_obs']}  n_used={extrap['n_cycles_used']}  "
          f"-out={extrap['n_outliers_removed']}  season_filter={extrap['season_type_filter']}")
    print(f"  series_end={str(extrap['series_end_date'])[:10]}  "
          f"forecast_EOS={str(extrap['forecast_eos_date'])[:10]}  "
          f"incerteza={extrap['uncertainty_days']}d")
    days_ahead = (pd.Timestamp(extrap['forecast_eos_date']) - pd.Timestamp(extrap['series_end_date'])).days
    print(f"  >>> EOS previsto {days_ahead}d após fim da série")
    print("  matched_cycles:")
    for m in extrap["matched_cycles"]:
        print(f"    ciclo {m['cycle_num']:2d} ({m['season_type']:12s})  "
              f"dist={m['distance']:.3f}  w={m['weight']:.3f}  dur={m['dur_days']:.0f}d")


if __name__ == "__main__":
    main()
