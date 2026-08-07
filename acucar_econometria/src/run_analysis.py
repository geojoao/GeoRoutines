"""
Estudo econometrico: o Brasil e tomador ou formulador do preco mundial do acucar?

Estrategia de identificacao
---------------------------
Duas hipoteses observacionalmente distintas:

  TOMADOR DE PRECO (small country): o preco mundial e exogeno ao Brasil. A
    producao brasileira REAGE ao preco (causalidade preco -> quantidade) e
    choques de oferta brasileiros NAO movem o preco mundial.

  FORMULADOR / PAIS GRANDE (market power): a producao brasileira desloca o
    preco mundial ao longo da demanda residual do resto do mundo. O sinal
    esperado e NEGATIVO (mais acucar brasileiro -> preco mundial menor) e a
    causalidade corre quantidade -> preco.

Bateria de testes: raiz unitaria (ADF/KPSS), cointegracao (Engle-Granger e
Johansen), VAR/VECM com causalidade de Granger nos dois sentidos, regressao de
"pais grande" (elasticidade do preco mundial a producao brasileira, com erros
HAC) e IRF/FEVD. Robustez com cana, preco nominal e subamostra pos-1990.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_coint_rank

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figuras"
OUT = ROOT / "outputs"
FIG.mkdir(parents=True, exist_ok=True)

R = {}  # dicionario de resultados


def load():
    df = pd.read_csv(DATA / "dataset_anual.csv")
    df = df.dropna(subset=["lp_real", "lq_acucar"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 1. Descritivas e participacao do Brasil
# --------------------------------------------------------------------------- #
def descritivas(df):
    R["periodo"] = [int(df.year.min()), int(df.year.max())]
    R["n_obs"] = int(len(df))
    R["share_brasil_inicio"] = float(df["share_brasil"].dropna().iloc[0])
    R["share_brasil_fim"] = float(df["share_brasil"].dropna().iloc[-1])
    R["share_brasil_media_ult10"] = float(df["share_brasil"].dropna().tail(10).mean())
    R["corr_lp_lq"] = float(df["lp_real"].corr(df["lq_acucar"]))

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(df.year, df.preco_real_usd_kg, color="#c0392b", lw=1.8)
    ax[0].set_ylabel("Preco real do acucar\n(US$/kg de 2023)")
    ax[0].set_title("Preco mundial do acucar (World Bank) x producao brasileira (FAOSTAT)")
    ax0b = ax[0].twinx()
    ax0b.plot(df.year, df.raw_sugar_t / 1e6, color="#2c3e50", lw=1.8, ls="--")
    ax0b.set_ylabel("Producao BR de acucar\n(milhoes t)", color="#2c3e50")
    ax[1].fill_between(df.year, 100 * df.share_brasil, color="#27ae60", alpha=.35)
    ax[1].plot(df.year, 100 * df.share_brasil, color="#1e8449", lw=1.8)
    ax[1].set_ylabel("Participacao do Brasil\nno acucar mundial (%)")
    ax[1].set_xlabel("Ano")
    fig.tight_layout()
    fig.savefig(FIG / "01_series_e_participacao.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Testes de raiz unitaria
# --------------------------------------------------------------------------- #
def _adf(x, name, reg="ct"):
    s = adfuller(x.dropna(), regression=reg, autolag="AIC")
    return {"var": name, "adf_stat": float(s[0]), "pvalue": float(s[1]),
            "lags": int(s[2]), "reg": reg}


def _kpss(x, name, reg="ct"):
    s = kpss(x.dropna(), regression=reg, nlags="auto")
    return {"var": name, "kpss_stat": float(s[0]), "pvalue": float(s[1]), "reg": reg}


def raiz_unitaria(df):
    res = {"adf": [], "kpss": []}
    series = {
        "lp_real (nivel)": (df.lp_real, "ct"),
        "lq_acucar (nivel)": (df.lq_acucar, "ct"),
        "d.lp_real": (df.lp_real.diff(), "c"),
        "d.lq_acucar": (df.lq_acucar.diff(), "c"),
    }
    for nm, (x, reg) in series.items():
        res["adf"].append(_adf(x, nm, reg))
        res["kpss"].append(_kpss(x, nm, reg))
    R["raiz_unitaria"] = res


# --------------------------------------------------------------------------- #
# 3. Cointegracao
# --------------------------------------------------------------------------- #
def cointegracao(df):
    y = df[["lp_real", "lq_acucar"]].dropna()
    # Engle-Granger nos dois sentidos
    eg1 = coint(y.lp_real, y.lq_acucar, trend="ct")
    eg2 = coint(y.lq_acucar, y.lp_real, trend="ct")
    # Johansen
    joh = coint_johansen(y.values, det_order=1, k_ar_diff=1)
    trace = joh.lr1.tolist()
    trace_cv = joh.cvt.tolist()  # 90/95/99
    R["cointegracao"] = {
        "engle_granger_lp_lq": {"stat": float(eg1[0]), "pvalue": float(eg1[1])},
        "engle_granger_lq_lp": {"stat": float(eg2[0]), "pvalue": float(eg2[1])},
        "johansen_trace_stats": trace,
        "johansen_trace_cv95": [row[1] for row in trace_cv],
        "johansen_r0_rejeita_95": bool(trace[0] > trace_cv[0][1]),
        "johansen_r1_rejeita_95": bool(trace[1] > trace_cv[1][1]),
    }


# --------------------------------------------------------------------------- #
# 4. VAR em diferencas + causalidade de Granger nos dois sentidos
# --------------------------------------------------------------------------- #
def var_granger(df, pcol="lp_real", qcol="lq_acucar", tag="base"):
    d = df[[pcol, qcol]].diff().dropna()
    d.columns = ["dp", "dq"]
    model = VAR(d)
    sel = model.select_order(maxlags=6)
    p = int(sel.aic) if sel.aic and sel.aic >= 1 else 1
    res = model.fit(p)

    # H0: a causante NAO Granger-causa a causada
    g_q_to_p = res.test_causality("dp", ["dq"], kind="f")  # producao -> preco
    g_p_to_q = res.test_causality("dq", ["dp"], kind="f")  # preco -> producao

    out = {
        "tag": tag, "lags_VAR": p,
        "granger_q_to_p": {"stat": float(g_q_to_p.test_statistic),
                            "pvalue": float(g_q_to_p.pvalue)},
        "granger_p_to_q": {"stat": float(g_p_to_q.test_statistic),
                           "pvalue": float(g_p_to_q.pvalue)},
    }

    if tag == "base":
        # IRF e FEVD: ordenacao dq -> dp (producao predeterminada no ano)
        res_ord = VAR(d[["dq", "dp"]]).fit(p)
        irf = res_ord.irf(10)
        fig = irf.plot(orth=True, impulse="dq", response="dp")
        fig.suptitle("Resposta do preco mundial a um choque na producao brasileira")
        fig.savefig(FIG / "02_irf_choque_producao_sobre_preco.png", dpi=130)
        plt.close(fig)

        fevd = res_ord.fevd(10)
        # fração da variancia do preco (indice 1) explicada pela producao (indice 0)
        R["fevd_preco_expl_por_producao_h10"] = float(fevd.decomp[1, -1, 0])
    return out


# --------------------------------------------------------------------------- #
# 5. Regressao de "pais grande": elasticidade do preco a producao brasileira
# --------------------------------------------------------------------------- #
def pais_grande(df):
    d = df[["lp_real", "lq_acucar", "lq_cana"]].diff().dropna()
    d.columns = ["dp", "dq", "dcana"]

    def hac_ols(y, X, L=2):
        X = sm.add_constant(X)
        return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})

    # (a) contemporaneo
    m1 = hac_ols(d.dp, d.dq)
    # (b) com defasagem (reduz simultaneidade): dp ~ dq_{t} + dq_{t-1}
    d2 = pd.DataFrame({"dp": d.dp, "dq": d.dq, "dq_l1": d.dq.shift(1)}).dropna()
    m2 = hac_ols(d2.dp, d2[["dq", "dq_l1"]])
    # (c) so defasada (dq predeterminada): dp ~ dq_{t-1}
    d3 = pd.DataFrame({"dp": d.dp, "dq_l1": d.dq.shift(1)}).dropna()
    m3 = hac_ols(d3.dp, d3[["dq_l1"]])

    R["pais_grande"] = {
        "contemporaneo": {"beta_dq": float(m1.params["dq"]),
                          "se": float(m1.bse["dq"]),
                          "t": float(m1.tvalues["dq"]),
                          "pvalue": float(m1.pvalues["dq"]),
                          "r2": float(m1.rsquared)},
        "com_defasagem_soma": {
            "beta_dq_contemp": float(m2.params["dq"]),
            "p_dq_contemp": float(m2.pvalues["dq"]),
            "beta_dq_l1": float(m2.params["dq_l1"]),
            "p_dq_l1": float(m2.pvalues["dq_l1"]),
            "soma_beta": float(m2.params["dq"] + m2.params["dq_l1"])},
        "so_defasada": {"beta_dq_l1": float(m3.params["dq_l1"]),
                        "p_dq_l1": float(m3.pvalues["dq_l1"])},
    }

    # scatter contemporaneo
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(100 * d.dq, 100 * d.dp, s=28, color="#2980b9", alpha=.75)
    xs = np.linspace(d.dq.min(), d.dq.max(), 50)
    ax.plot(100 * xs, 100 * (m1.params["const"] + m1.params["dq"] * xs),
            color="#c0392b", lw=2)
    ax.axhline(0, color="grey", lw=.6); ax.axvline(0, color="grey", lw=.6)
    ax.set_xlabel("Variacao anual da producao BR de acucar (%)")
    ax.set_ylabel("Variacao anual do preco mundial real (%)")
    ax.set_title(f"Elasticidade preco-producao: {m1.params['dq']:.2f}  (p={m1.pvalues['dq']:.3f})")
    fig.tight_layout()
    fig.savefig(FIG / "03_elasticidade_preco_producao.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Robustez
# --------------------------------------------------------------------------- #
def robustez(df):
    rob = {}
    rob["cana"] = var_granger(df, "lp_real", "lq_cana", tag="cana")
    rob["preco_nominal"] = var_granger(df, "lp_nom", "lq_acucar", tag="nominal")
    sub = df[df.year >= 1990].reset_index(drop=True)
    rob["pos_1990"] = var_granger(sub, "lp_real", "lq_acucar", tag="pos1990")
    R["robustez"] = rob


def toda_yamamoto(df, pcol="lp_real", qcol="lq_acucar"):
    """
    Causalidade de Granger a la Toda-Yamamoto (1995): VAR em NIVEL com p+dmax
    defasagens e teste de Wald sobre as p primeiras. Valido mesmo com series de
    ordens de integracao diferentes / cointegradas (aqui preco ~I(0)/tend,
    producao ~I(1)), contornando a ambiguidade dos testes de raiz unitaria.
    """
    lv = df[[pcol, qcol]].rename(columns={pcol: "p", qcol: "q"}).reset_index(drop=True)
    p = int(VAR(lv).select_order(maxlags=6).aic) or 1
    dmax = 1
    k = p + dmax

    Y = lv.copy()
    for i in range(1, k + 1):
        Y[f"p_l{i}"] = Y["p"].shift(i)
        Y[f"q_l{i}"] = Y["q"].shift(i)
    Y["trend"] = np.arange(len(Y))
    Y = Y.dropna().reset_index(drop=True)

    def eq(dep):
        exog = ["trend"] + [f"p_l{i}" for i in range(1, k + 1)] + [f"q_l{i}" for i in range(1, k + 1)]
        X = sm.add_constant(Y[exog])
        return sm.OLS(Y[dep], X).fit(cov_type="HC1"), exog

    # p -> q : testa q_dep sobre p_l1..p_lp (exclui a defasagem extra dmax)
    m_q, _ = eq("q")
    r_p_to_q = m_q.f_test([f"p_l{i} = 0" for i in range(1, p + 1)])
    # q -> p
    m_p, _ = eq("p")
    r_q_to_p = m_p.f_test([f"q_l{i} = 0" for i in range(1, p + 1)])

    R["toda_yamamoto"] = {
        "p_lags": p, "dmax": dmax,
        "granger_q_to_p": {"stat": float(np.ravel(r_q_to_p.fvalue)[0]),
                           "pvalue": float(r_q_to_p.pvalue)},
        "granger_p_to_q": {"stat": float(np.ravel(r_p_to_q.fvalue)[0]),
                           "pvalue": float(r_p_to_q.pvalue)},
    }


def main():
    df = load()
    descritivas(df)
    raiz_unitaria(df)
    cointegracao(df)
    R["var_granger_base"] = var_granger(df, tag="base")
    toda_yamamoto(df)
    pais_grande(df)
    robustez(df)

    (OUT / "resultados.json").write_text(json.dumps(R, indent=2, ensure_ascii=False))
    print(json.dumps(R, indent=2, ensure_ascii=False))
    print("\nFiguras e resultados salvos em outputs/")


if __name__ == "__main__":
    main()
