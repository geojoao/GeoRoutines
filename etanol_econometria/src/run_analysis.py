"""
Estudo econometrico: o Brasil e tomador ou formulador do preco mundial do ETANOL?

Diferente do acucar, o etanol NAO tem um mercado mundial unico e liquido: e um
mercado regionalmente segmentado (milho nos EUA, cana no Brasil, UE), com o
preco de referencia mundial praticamente ancorado no preco americano (o "corn
belt"). O Brasil, que foi quase-monopolista nos anos 1980 (era Proalcool), hoje
responde por ~30% da producao mundial, atras dos EUA (~50%).

Mesma estrategia de identificacao do estudo do acucar:
  TOMADOR: preco mundial exogeno; producao BR reage ao preco (preco->producao);
    choques de oferta brasileiros NAO movem o preco mundial.
  FORMULADOR/PAIS GRANDE: producao BR desloca o preco mundial (sinal NEGATIVO;
    producao->preco).

Bateria: raiz unitaria (ADF/KPSS), cointegracao (Engle-Granger e Johansen),
causalidade de Granger (VAR em diferencas e Toda-Yamamoto), regressao de "pais
grande" (elasticidade preco-producao, erros HAC) e IRF/FEVD. Como o etanol
brasileiro vive do arbitragem com o ACUCAR (mesma cana) e com a GASOLINA/PETROLEO
(no bico da bomba flex), acrescenta-se um bloco de "nexo": a producao brasileira
de etanol responde ao preco do acucar e do petroleo?

Amostra: anual 1990-2022 (n=33; preco = OECD-FAO, so historico).
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA, OUT, FIG = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
R = {}
MAXLAG = 3  # amostra curta (n=33) -> parcimonia


def load():
    return pd.read_csv(DATA / "dataset_anual.csv").dropna(subset=["lp_real", "lq"]).reset_index(drop=True)


def descritivas(df):
    R["periodo"] = [int(df.year.min()), int(df.year.max())]
    R["n_obs"] = int(len(df))
    R["share_brasil_inicio"] = float(df["share_brasil"].iloc[0])
    R["share_brasil_fim"] = float(df["share_brasil"].iloc[-1])
    R["corr_preco_ref_mundo_vs_eua"] = float(
        df[["preco_etanol_nom_usd_hl", "eth_usa_usd_hl"]].dropna().corr().iloc[0, 1])
    R["corr_lp_lq"] = float(df["lp_real"].corr(df["lq"]))

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(df.year, df.preco_etanol_real_usd_hl, color="#8e44ad", lw=1.8, label="Preco mundial etanol (real)")
    ax[0].set_ylabel("Preco real do etanol\n(US$/hl de 2022)")
    ax[0].set_title("Preco mundial do etanol (OECD-FAO) x producao brasileira (EIA)")
    axb = ax[0].twinx()
    axb.plot(df.year, df.prod_brasil_tbpd, color="#16a085", lw=1.8, ls="--")
    axb.set_ylabel("Producao BR de etanol\n(mil barris/dia)", color="#16a085")
    ax[1].fill_between(df.year, 100 * df.share_brasil, color="#16a085", alpha=.30)
    ax[1].plot(df.year, 100 * df.share_brasil, color="#0e6655", lw=1.8, label="Brasil")
    ax[1].plot(df.year, 100 * df.prod_eua_tbpd / df.prod_mundo_tbpd, color="#c0392b", lw=1.6, ls=":", label="EUA")
    ax[1].set_ylabel("Participacao na producao\nmundial de etanol (%)")
    ax[1].set_xlabel("Ano"); ax[1].legend(loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "01_series_e_participacao.png", dpi=130); plt.close(fig)


def _adf(x, name, reg): s = adfuller(x.dropna(), regression=reg, autolag="AIC"); return {"var": name, "adf_stat": float(s[0]), "pvalue": float(s[1]), "lags": int(s[2])}
def _kpss(x, name, reg): s = kpss(x.dropna(), regression=reg, nlags="auto"); return {"var": name, "kpss_stat": float(s[0]), "pvalue": float(s[1])}


def raiz_unitaria(df):
    res = {"adf": [], "kpss": []}
    for nm, x, reg in [("lp_real (nivel)", df.lp_real, "ct"), ("lq (nivel)", df.lq, "ct"),
                        ("d.lp_real", df.lp_real.diff(), "c"), ("d.lq", df.lq.diff(), "c")]:
        res["adf"].append(_adf(x, nm, reg)); res["kpss"].append(_kpss(x, nm, reg))
    R["raiz_unitaria"] = res


def cointegracao(df):
    y = df[["lp_real", "lq"]].dropna()
    eg1 = coint(y.lp_real, y.lq, trend="ct"); eg2 = coint(y.lq, y.lp_real, trend="ct")
    joh = coint_johansen(y.values, det_order=1, k_ar_diff=1)
    R["cointegracao"] = {
        "engle_granger_lp_lq_pvalue": float(eg1[1]),
        "engle_granger_lq_lp_pvalue": float(eg2[1]),
        "johansen_trace": joh.lr1.tolist(),
        "johansen_cv95": [row[1] for row in joh.cvt.tolist()],
        "johansen_r0_rejeita_95": bool(joh.lr1[0] > joh.cvt[0][1]),
        "johansen_r1_rejeita_95": bool(joh.lr1[1] > joh.cvt[1][1]),
    }


def var_granger(df, pcol, qcol, tag, do_irf=False):
    d = df[[pcol, qcol]].diff().dropna(); d.columns = ["dp", "dq"]
    model = VAR(d)
    sel = model.select_order(maxlags=MAXLAG)
    p = int(sel.aic) if sel.aic and sel.aic >= 1 else 1
    res = model.fit(p)
    gq = res.test_causality("dp", ["dq"], kind="f")  # producao -> preco
    gp = res.test_causality("dq", ["dp"], kind="f")  # preco -> producao
    out = {"tag": tag, "lags_VAR": p,
           "granger_q_to_p": {"stat": float(gq.test_statistic), "pvalue": float(gq.pvalue)},
           "granger_p_to_q": {"stat": float(gp.test_statistic), "pvalue": float(gp.pvalue)}}
    if do_irf:
        res_o = VAR(d[["dq", "dp"]]).fit(p)
        fig = res_o.irf(10).plot(orth=True, impulse="dq", response="dp")
        fig.suptitle("Resposta do preco mundial do etanol a um choque na producao brasileira")
        fig.savefig(FIG / "02_irf_choque_producao_sobre_preco.png", dpi=130); plt.close(fig)
        R["fevd_preco_expl_por_producao_h10"] = float(res_o.fevd(10).decomp[1, -1, 0])
    return out


def toda_yamamoto(df, pcol="lp_real", qcol="lq"):
    lv = df[[pcol, qcol]].rename(columns={pcol: "p", qcol: "q"}).reset_index(drop=True)
    p = int(VAR(lv).select_order(maxlags=MAXLAG).aic) or 1
    k = p + 1
    Y = lv.copy()
    for i in range(1, k + 1):
        Y[f"p_l{i}"] = Y["p"].shift(i); Y[f"q_l{i}"] = Y["q"].shift(i)
    Y["trend"] = np.arange(len(Y)); Y = Y.dropna().reset_index(drop=True)
    exog = ["trend"] + [f"p_l{i}" for i in range(1, k + 1)] + [f"q_l{i}" for i in range(1, k + 1)]

    def eq(dep): return sm.OLS(Y[dep], sm.add_constant(Y[exog])).fit(cov_type="HC1")
    r_q2p = eq("p").f_test([f"q_l{i} = 0" for i in range(1, p + 1)])
    r_p2q = eq("q").f_test([f"p_l{i} = 0" for i in range(1, p + 1)])
    R["toda_yamamoto"] = {"p_lags": p,
        "granger_q_to_p": {"stat": float(np.ravel(r_q2p.fvalue)[0]), "pvalue": float(r_q2p.pvalue)},
        "granger_p_to_q": {"stat": float(np.ravel(r_p2q.fvalue)[0]), "pvalue": float(r_p2q.pvalue)}}


def pais_grande(df):
    d = df[["lp_real", "lq"]].diff().dropna(); d.columns = ["dp", "dq"]
    def hac(y, X, L=2): return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    m1 = hac(d.dp, d.dq)
    d2 = pd.DataFrame({"dp": d.dp, "dq": d.dq, "dq_l1": d.dq.shift(1)}).dropna()
    m2 = hac(d2.dp, d2[["dq", "dq_l1"]])
    R["pais_grande"] = {
        "contemporaneo": {"beta_dq": float(m1.params["dq"]), "pvalue": float(m1.pvalues["dq"]), "r2": float(m1.rsquared)},
        "com_defasagem": {"beta_dq": float(m2.params["dq"]), "p_dq": float(m2.pvalues["dq"]),
                          "beta_dq_l1": float(m2.params["dq_l1"]), "p_dq_l1": float(m2.pvalues["dq_l1"]),
                          "soma": float(m2.params["dq"] + m2.params["dq_l1"])}}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(100 * d.dq, 100 * d.dp, s=30, color="#8e44ad", alpha=.75)
    xs = np.linspace(d.dq.min(), d.dq.max(), 50)
    ax.plot(100 * xs, 100 * (m1.params["const"] + m1.params["dq"] * xs), color="#c0392b", lw=2)
    ax.axhline(0, color="grey", lw=.6); ax.axvline(0, color="grey", lw=.6)
    ax.set_xlabel("Variacao anual da producao BR de etanol (%)")
    ax.set_ylabel("Variacao anual do preco mundial real do etanol (%)")
    ax.set_title(f"Elasticidade preco-producao: {m1.params['dq']:.2f}  (p={m1.pvalues['dq']:.3f})")
    fig.tight_layout(); fig.savefig(FIG / "03_elasticidade_preco_producao.png", dpi=130); plt.close(fig)


def nexo(df):
    """Producao BR de etanol responde ao preco do acucar e do petroleo?"""
    d = df[["lq", "lp_acucar", "lp_oil", "lp_real"]].diff().dropna()
    d.columns = ["dq", "dsugar", "doil", "dp_et"]
    def hac(y, X, L=2): return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    # producao BR ~ preco acucar (defasado) + petroleo (defasado)
    dd = pd.DataFrame({"dq": d.dq, "dsugar_l1": d.dsugar.shift(1), "doil_l1": d.doil.shift(1),
                       "dp_et_l1": d.dp_et.shift(1)}).dropna()
    m = hac(dd.dq, dd[["dsugar_l1", "doil_l1", "dp_et_l1"]])
    R["nexo_producao_br"] = {
        "beta_acucar_l1": float(m.params["dsugar_l1"]), "p_acucar_l1": float(m.pvalues["dsugar_l1"]),
        "beta_oil_l1": float(m.params["doil_l1"]), "p_oil_l1": float(m.pvalues["doil_l1"]),
        "beta_preco_etanol_l1": float(m.params["dp_et_l1"]), "p_preco_etanol_l1": float(m.pvalues["dp_et_l1"]),
        "r2": float(m.rsquared)}
    # quem explica o preco mundial do etanol: choque brasileiro vs petroleo (FEVD 3-var)
    v = df[["lq", "lp_oil", "lp_real"]].diff().dropna(); v.columns = ["dq", "doil", "dp"]
    resv = VAR(v).fit(1)
    fevd = resv.fevd(10).decomp[:, -1, :]  # linhas=variavel resposta, colunas=choque
    R["fevd_preco_etanol_h10"] = {"por_producao_br": float(fevd[2, 0]),
                                  "por_petroleo": float(fevd[2, 1]),
                                  "por_proprio": float(fevd[2, 2])}


def robustez(df):
    rob = {}
    rob["preco_nominal"] = var_granger(df, "lp_nom", "lq", "nominal")
    sub = df[df.year >= 2005].reset_index(drop=True)  # era pos-RFS / Brasil ja #2
    rob["pos_2005"] = var_granger(sub, "lp_real", "lq", "pos2005")
    R["robustez"] = rob


def main():
    df = load()
    descritivas(df); raiz_unitaria(df); cointegracao(df)
    R["var_granger_base"] = var_granger(df, "lp_real", "lq", "base", do_irf=True)
    toda_yamamoto(df); pais_grande(df); nexo(df); robustez(df)
    (OUT / "resultados.json").write_text(json.dumps(R, indent=2, ensure_ascii=False))
    print(json.dumps(R, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
