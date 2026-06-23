"""
Script de debugging para diagnóstico de hexágonos sem detecção de pico vegetativo.

Para cada hexágono problemático (tem EVI mas sem fenologia), executa o pipeline
passo a passo e registra exatamente onde a detecção falha.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from fenologia.phenophase import (
    adaptive_smoothing,
    detect_vegetation_peaks,
    segment_around_peaks,
    fit_gaussian_to_cycle,
    double_logistic,
)

# Thresholds de produção (extract_phenology.py)
MIN_CYCLE_DAYS = {
    "soja": 120, "cana": 150, "arroz": 100, "algodao": 110,
    "cafe": 120, "citrus": 120, "dende": 120,
    "outras_lavouras_temporarias": 100, "outras_lavouras_perenes": 120,
    "segunda_safra": 90, "segunda_safra_algodao": 100,
    "segunda_safra_outras_temporarias": 90,
}
MAX_CYCLE_DAYS = {
    "soja": 180, "cana": 420, "arroz": 160, "algodao": 220,
    "cafe": 400, "citrus": 400, "dende": 400,
    "outras_lavouras_temporarias": 200, "outras_lavouras_perenes": 400,
    "segunda_safra": 150, "segunda_safra_algodao": 220,
    "segunda_safra_outras_temporarias": 180,
}
MIN_EVI_AMPLITUDE = 0.08
MIN_R2_PER_CYCLE = 0.80
MIN_GROWING_DAYS = 35

OUTPUT_DIR = Path("/tmp/claude-0/-home-user-GeoRoutines/26308b7a-b490-5a76-b07b-0a1edd131b59/scratchpad")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _growing_days(cycle: dict) -> float:
    pd_ = cycle["phenophase_days"]
    return pd_["eos_days"] - pd_["sos_days"]


def diagnose_hexagono(hex_id: str, dates: np.ndarray, evi: np.ndarray,
                       cultura: str, ax_row=None) -> dict:
    """
    Roda o pipeline passo a passo e retorna diagnóstico detalhado.
    """
    result = {
        "hex_id": hex_id,
        "cultura": cultura,
        "n_pontos": int(mask := ~np.isnan(evi), mask.sum())[1] if False else None,
        "falha_em": None,
        "detalhes": {},
        "ciclos_tentados": [],
        "ciclos_aceitos": [],
    }

    # Filtra NaN
    mask = ~np.isnan(evi)
    result["n_pontos"] = int(mask.sum())
    result["evi_max"] = float(np.nanmax(evi))
    result["evi_std"] = float(np.nanstd(evi))

    if mask.sum() < 10:
        result["falha_em"] = "dados_insuficientes"
        result["detalhes"]["n_validos"] = int(mask.sum())
        return result

    dates_v = dates[mask]
    evi_v = evi[mask].astype(float)

    # Etapa 1: Suavização
    evi_smooth = adaptive_smoothing(evi_v, dates_v, method="both")
    result["detalhes"]["n_pontos_validos"] = int(mask.sum())

    # Etapa 2: Detecção de picos
    total_days = float((dates_v[-1] - dates_v[0]) / np.timedelta64(1, "D"))
    min_dist = MIN_CYCLE_DAYS.get(cultura, 90)
    # Distância de detecção = 65% do ciclo mínimo (igual ao extract_phenometrics)
    detection_dist = max(60, int(min_dist * 0.65))
    ndvi_std = float(np.std(evi_smooth))
    prominence_min = max(0.03, ndvi_std * 0.25)
    min_distance_idx = max(1, int(detection_dist * len(evi_smooth) / total_days))
    min_distance_idx_full = max(1, int(min_dist * len(evi_smooth) / total_days))

    peaks_raw, peak_props = find_peaks(
        evi_smooth,
        distance=min_distance_idx,
        prominence=prominence_min,
    )

    # Testa sem filtro de distância (para ver picos bloqueados)
    peaks_nodist, _ = find_peaks(evi_smooth, prominence=prominence_min)
    peaks_noprom, _ = find_peaks(evi_smooth, distance=min_distance_idx_full)
    # Picos adicionais capturados pela distância reduzida
    peaks_full_dist, _ = find_peaks(evi_smooth, distance=min_distance_idx_full, prominence=prominence_min)

    result["detalhes"]["prominence_threshold"] = round(prominence_min, 4)
    result["detalhes"]["min_distance_dias"] = detection_dist
    result["detalhes"]["min_distance_idx"] = min_distance_idx
    result["detalhes"]["n_picos_detectados"] = int(len(peaks_raw))
    result["detalhes"]["n_picos_sem_distancia"] = int(len(peaks_nodist))
    result["detalhes"]["n_picos_distancia_120"] = int(len(peaks_full_dist))
    result["detalhes"]["n_picos_distancia_78"] = int(len(peaks_raw))

    if len(peaks_raw) == 0:
        if len(peaks_nodist) > 0:
            result["falha_em"] = "pico_bloqueado_por_distancia"
        else:
            result["falha_em"] = "sem_pico_suficiente_prominence"
        result["detalhes"]["evi_std"] = round(ndvi_std, 4)
    else:
        # Etapa 3-4: Segmentação + Gaussian fit
        cycles = segment_around_peaks(evi_v, dates_v, peaks_raw)
        for cyc in cycles:
            fit = fit_gaussian_to_cycle(evi_v, dates_v, cyc, quality_threshold=0.6)
            cyc_diag = {
                "cycle_num": cyc["cycle_num"],
                "length_days": round(float(cyc["length_days"]), 1),
                "fit_success_06": fit.get("fit_success", False),
                "r2": round(float(fit.get("r_squared", 0.0)), 4) if fit.get("fit_success") else None,
                "amplitude": None,
                "growing_days": None,
                "rejeitado_por": [],
            }
            if fit.get("fit_success"):
                amp = fit["gaussian_params"]["amplitude"]
                grow = _growing_days(fit)
                max_days = MAX_CYCLE_DAYS.get(cultura, 400)
                cyc_diag["amplitude"] = round(float(amp), 4)
                cyc_diag["growing_days"] = round(float(grow), 1)

                motivos = []
                if fit["r_squared"] < MIN_R2_PER_CYCLE:
                    motivos.append(f"R²={fit['r_squared']:.3f} < {MIN_R2_PER_CYCLE}")
                if amp < MIN_EVI_AMPLITUDE:
                    motivos.append(f"amplitude={amp:.3f} < {MIN_EVI_AMPLITUDE}")
                if grow < MIN_GROWING_DAYS:
                    motivos.append(f"growing_days={grow:.0f} < {MIN_GROWING_DAYS}")
                if grow > max_days:
                    motivos.append(f"growing_days={grow:.0f} > {max_days}")

                cyc_diag["rejeitado_por"] = motivos
                if not motivos:
                    result["ciclos_aceitos"].append(cyc_diag)
            else:
                cyc_diag["rejeitado_por"] = [fit.get("reason", "fit falhou")]
            result["ciclos_tentados"].append(cyc_diag)
            result["ciclos_tentados"][-1]["fit_obj"] = fit
            result["ciclos_tentados"][-1]["cycle_obj"] = cyc

        if not result["ciclos_aceitos"]:
            all_rejections = []
            for c in result["ciclos_tentados"]:
                all_rejections.extend(c["rejeitado_por"])
            # Agrupa o motivo mais comum
            from collections import Counter
            counter = Counter(
                r.split("=")[0].split("<")[0].split(">")[0].strip()
                for r in all_rejections
            )
            result["falha_em"] = "ciclo_rejeitado_filtros"
            result["detalhes"]["motivos_rejeicao"] = all_rejections
            result["detalhes"]["motivo_principal"] = counter.most_common(1)[0][0] if counter else "?"
        else:
            result["falha_em"] = None  # Hexágono funcionou!

    # Guarda dados para plotar
    result["_evi_v"] = evi_v
    result["_dates_v"] = dates_v
    result["_evi_smooth"] = evi_smooth
    result["_peaks_raw"] = peaks_raw
    result["_peaks_nodist"] = peaks_nodist

    return result


def plot_diagnostic(diag: dict, titulo: str, output_path: Path):
    """Gera plot diagnóstico de 3 painéis para um hexágono problemático."""
    evi_v = diag["_evi_v"]
    dates_v = diag["_dates_v"]
    evi_smooth = diag["_evi_smooth"]
    peaks = diag["_peaks_raw"]
    peaks_nodist = diag["_peaks_nodist"]

    # Cores para ciclos
    COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

    # ── Painel 1: EVI bruto + suavizado + picos detectados ──────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates_v, evi_v, "k-", lw=1, alpha=0.4, label="EVI bruto")
    ax1.plot(dates_v, evi_smooth, "b-", lw=2, label="EVI suavizado")

    if len(peaks) > 0:
        ax1.scatter(dates_v[peaks], evi_smooth[peaks], c="green", s=120,
                    zorder=5, marker="^", label=f"Picos detectados ({len(peaks)})")
    if len(peaks_nodist) > len(peaks):
        extras = [p for p in peaks_nodist if p not in peaks]
        ax1.scatter(dates_v[extras], evi_smooth[extras], c="orange", s=80,
                    zorder=4, marker="^", alpha=0.6,
                    label=f"Picos bloq. por distância ({len(extras)})")

    # Linha de prominence threshold
    prom = diag["detalhes"].get("prominence_threshold", 0.03)
    ax1.axhline(np.mean(evi_smooth) + prom, ls=":", color="gray", lw=1,
                label=f"Prominence min = {prom:.3f}")

    ax1.set_ylabel("EVI", fontsize=11)
    ax1.set_title(f"[{diag['hex_id']}] {diag['cultura'].upper()}  |  "
                  f"Pontos válidos: {diag['n_pontos']}  |  "
                  f"EVI max: {diag['evi_max']:.3f}  |  "
                  f"Std: {diag['evi_std']:.3f}", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Anotação do motivo de falha
    falha = diag.get("falha_em", "ok")
    motivos = diag["detalhes"].get("motivos_rejeicao", [])
    label_falha = f"Falha: {falha}"
    if motivos:
        label_falha += "\n" + " | ".join(motivos[:3])
    ax1.text(0.01, 0.05, label_falha, transform=ax1.transAxes,
             fontsize=9, color="red", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Painel 2: Janelas de ciclo segmentadas ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates_v, evi_v, "k-", lw=0.8, alpha=0.3)
    ax2.plot(dates_v, evi_smooth, "b-", lw=1.5, alpha=0.7)

    for i, cyc_diag in enumerate(diag.get("ciclos_tentados", [])):
        cyc = cyc_diag.get("cycle_obj", {})
        if not cyc:
            continue
        s, e = int(cyc["start_idx"]), int(cyc["end_idx"])
        color = COLORS[i % len(COLORS)]
        aceito = not cyc_diag["rejeitado_por"]
        alpha = 0.25 if aceito else 0.12
        ax2.axvspan(dates_v[s], dates_v[e], alpha=alpha, color=color)
        ax2.axvline(dates_v[s], ls="--", lw=0.8, color=color, alpha=0.6)
        ax2.axvline(dates_v[e], ls="--", lw=0.8, color=color, alpha=0.6)
        label = f"Ciclo {cyc_diag['cycle_num']} ({cyc_diag['length_days']:.0f}d)"
        if cyc_diag.get("r2") is not None:
            label += f"\nR²={cyc_diag['r2']:.3f}"
        if cyc_diag["rejeitado_por"]:
            label += f"\n✗ {'; '.join(cyc_diag['rejeitado_por'][:2])}"
        else:
            label += "\n✓ ACEITO"
        mid_idx = (s + e) // 2
        ax2.text(dates_v[mid_idx], ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else np.max(evi_smooth) * 0.95,
                 label, ha="center", va="top", fontsize=7.5, color=color,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax2.set_ylabel("EVI", fontsize=11)
    ax2.set_title("Segmentação em Ciclos — Motivos de Rejeição", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # ── Painel 3: Ajuste Gaussiano por ciclo ────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dates_v, evi_v, "k-", lw=1.2, alpha=0.5, label="EVI bruto")

    any_fit = False
    for i, cyc_diag in enumerate(diag.get("ciclos_tentados", [])):
        fit = cyc_diag.get("fit_obj", {})
        cyc = cyc_diag.get("cycle_obj", {})
        if not cyc or not fit.get("fit_success"):
            continue
        color = COLORS[i % len(COLORS)]
        s, e = int(cyc["start_idx"]), int(cyc["end_idx"])
        dates_cyc = dates_v[s:e + 1]
        days_from_start = np.array(
            [(d - dates_cyc[0]) / np.timedelta64(1, "D") for d in dates_cyc], dtype=float
        )
        p = fit["gaussian_params"]
        gauss_vals = double_logistic(days_from_start, p["amplitude"], p["m1"], p["k1"], p["m2"], p["k2"], p["offset"])
        label_g = (f"Ciclo {cyc_diag['cycle_num']} "
                   f"R²={fit['r_squared']:.3f} | "
                   f"amp={p['amplitude']:.3f} | "
                   f"grow={_growing_days(fit):.0f}d")
        ls = "-" if not cyc_diag["rejeitado_por"] else "--"
        ax3.plot(dates_cyc, gauss_vals, ls, lw=2.5, color=color, label=label_g)

        # Marca SOS/POS/EOS
        if "phenophase_dates" in fit:
            sos = fit["phenophase_dates"]["sos"]
            pos = fit["phenophase_dates"]["pos"]
            eos = fit["phenophase_dates"]["eos"]
            ax3.axvline(sos, ls=":", lw=1, color=color, alpha=0.7)
            ax3.axvline(pos, ls="-", lw=1.5, color=color, alpha=0.9)
            ax3.axvline(eos, ls=":", lw=1, color=color, alpha=0.7)
        any_fit = True

    if not any_fit:
        ax3.text(0.5, 0.5, "Nenhum ajuste Gaussiano bem-sucedido\n(R² < 0.60 ou fit divergiu)",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=13, color="red",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#ffe0e0", alpha=0.9))

    ax3.set_xlabel("Data", fontsize=11)
    ax3.set_ylabel("EVI", fontsize=11)
    ax3.set_title("Ajuste Gaussiano (tracejado = rejeitado pelos filtros de produção)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    if any_fit:
        ax3.legend(loc="upper right", fontsize=8)

    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {output_path.name}")


def find_problema_hexagonos(cultura: str = "soja", n_casos: int = 12):
    """
    Encontra hexágonos com EVI de uma cultura mas sem fenologia detectada.
    Retorna os mais interessantes (EVI alto, muitos pontos válidos).
    """
    print(f"\n{'='*60}")
    print(f"  Carregando dados para: {cultura}")
    print(f"{'='*60}")

    evi_df = pd.read_parquet("data/output/evi_brazil.parquet")
    feno_df = pd.read_parquet("data/output/fenologia_brasil.parquet")

    col_evi = f"evi_medio_{cultura}"
    if col_evi not in evi_df.columns:
        print(f"Coluna {col_evi} não encontrada!")
        return None, None

    # Hexágonos com EVI para esta cultura
    evi_cultura = evi_df[["id_hexagono", "data", col_evi]].copy()
    evi_cultura = evi_cultura.dropna(subset=[col_evi])

    stats = (
        evi_cultura
        .groupby("id_hexagono")[col_evi]
        .agg(n_validos="count", evi_max="max", evi_std="std", evi_mean="mean")
        .reset_index()
    )

    hex_com_evi = set(stats["id_hexagono"])
    hex_com_feno = set(
        feno_df[feno_df["cultura"] == cultura]["id_hexagono"].unique()
    )

    ausentes = hex_com_evi - hex_com_feno
    presentes = hex_com_evi & hex_com_feno

    print(f"  Hexágonos com EVI de {cultura}:       {len(hex_com_evi):>6,}")
    print(f"  Hexágonos com fenologia de {cultura}: {len(hex_com_feno):>6,}")
    print(f"  Hexágonos AUSENTES (problemas):       {len(ausentes):>6,}")
    print(f"  Taxa de detecção:                     {len(hex_com_feno)/len(hex_com_evi)*100:.1f}%")

    # Seleciona casos mais interessantes: EVI alto + muitos pontos
    stats_ausentes = stats[stats["id_hexagono"].isin(ausentes)].copy()
    stats_ausentes["score"] = (
        stats_ausentes["evi_max"].rank(pct=True) * 0.5 +
        stats_ausentes["n_validos"].rank(pct=True) * 0.3 +
        stats_ausentes["evi_std"].rank(pct=True) * 0.2
    )
    top_ausentes = stats_ausentes.nlargest(n_casos, "score")

    # Também pega alguns que funcionaram para comparação
    stats_presentes = stats[stats["id_hexagono"].isin(presentes)].copy()
    top_presentes = stats_presentes.nlargest(3, "evi_max")

    print(f"\n  Top {n_casos} problemáticos (score = EVI alto + muitos pontos):")
    print(top_ausentes[["id_hexagono", "n_validos", "evi_max", "evi_std"]].to_string(index=False))

    return evi_df, top_ausentes, top_presentes, feno_df


def run_debug(cultura: str = "soja", n_casos: int = 8):
    result = find_problema_hexagonos(cultura, n_casos)
    if result[0] is None:
        return

    evi_df, top_ausentes, top_presentes, feno_df = result
    col_evi = f"evi_medio_{cultura}"

    # ── Diagnóstico dos casos problemáticos ─────────────────────────────────
    print(f"\n{'='*60}")
    print("  DIAGNÓSTICO DETALHADO")
    print(f"{'='*60}")

    all_diags = []
    resumo_falhas = {}

    for _, row in top_ausentes.iterrows():
        hex_id = row["id_hexagono"]
        grp = evi_df[evi_df["id_hexagono"] == hex_id].sort_values("data")
        dates = grp["data"].values
        evi = grp[col_evi].values.astype(float)

        diag = diagnose_hexagono(hex_id, dates, evi, cultura)
        all_diags.append(diag)

        falha = diag.get("falha_em", "ok")
        resumo_falhas[falha] = resumo_falhas.get(falha, 0) + 1

        print(f"\n  [{hex_id}]  n={diag['n_pontos']}  EVI_max={diag['evi_max']:.3f}  std={diag['evi_std']:.3f}")
        print(f"    Falha em: {falha}")
        det = diag["detalhes"]
        if "prominence_threshold" in det:
            print(f"    Prominence threshold: {det['prominence_threshold']:.4f}")
            print(f"    Picos detectados: {det['n_picos_detectados']}  sem dist: {det['n_picos_sem_distancia']}")
        if "motivos_rejeicao" in det:
            for m in det["motivos_rejeicao"]:
                print(f"    ✗ {m}")

    print(f"\n  {'─'*40}")
    print(f"  RESUMO DE FALHAS ({cultura}):")
    for motivo, contagem in sorted(resumo_falhas.items(), key=lambda x: -x[1]):
        motivo_str = str(motivo) if motivo is not None else "ok (detectado)"
        print(f"    {motivo_str:<40} {contagem:>3}x")

    # ── Geração de plots ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  GERANDO PLOTS DE DIAGNÓSTICO")
    print(f"{'='*60}")

    # Agrupa: até 3 casos por página para não ficar muito pesado
    n_por_pag = 3
    paginas = [all_diags[i:i + n_por_pag] for i in range(0, len(all_diags), n_por_pag)]

    for pg_idx, pagina in enumerate(paginas):
        for d in pagina:
            out_path = OUTPUT_DIR / f"debug_{cultura}_{d['hex_id']}.png"
            titulo = (f"DEBUG {cultura.upper()} — {d['hex_id']} "
                      f"(falha: {d.get('falha_em', 'ok')})")
            plot_diagnostic(d, titulo, out_path)

    # ── Casos que funcionaram (para comparação) ─────────────────────────────
    print(f"\n  Gerando comparação com hexágonos que FUNCIONARAM...")
    for _, row in top_presentes.iterrows():
        hex_id = row["id_hexagono"]
        grp = evi_df[evi_df["id_hexagono"] == hex_id].sort_values("data")
        dates = grp["data"].values
        evi = grp[col_evi].values.astype(float)
        diag = diagnose_hexagono(hex_id, dates, evi, cultura)

        # Informa fenologia detectada
        feno_hex = feno_df[(feno_df["id_hexagono"] == hex_id) & (feno_df["cultura"] == cultura)]
        n_ciclos_feno = int(feno_hex["n_ciclos"].sum()) if len(feno_hex) > 0 else 0
        r2 = float(feno_hex["r2_medio"].mean()) if len(feno_hex) > 0 else 0
        titulo = (f"REFERÊNCIA (OK) {cultura.upper()} — {hex_id} "
                  f"| {n_ciclos_feno} ciclos detectados | R²={r2:.3f}")
        out_path = OUTPUT_DIR / f"debug_{cultura}_{hex_id}_OK.png"
        plot_diagnostic(diag, titulo, out_path)

    print(f"\n  Todos os plots salvos em: {OUTPUT_DIR}")
    return all_diags


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultura", default="soja")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    run_debug(cultura=args.cultura, n_casos=args.n)
