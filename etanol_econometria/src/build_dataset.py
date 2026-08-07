"""
Constroi o painel anual do estudo do etanol.

Junta:
  - preco MUNDIAL de referencia do etanol (OECD-FAO Agricultural Outlook /
    Aglink-Cosimo, "World ethanol reference price", USD por hectolitro) -> so a
    parte historica (1990-2022; 2023+ sao projecoes e sao descartadas)
  - producao brasileira de etanol (EIA International Energy Data, mil barris/dia)
  - producao mundial e dos EUA de etanol (EIA) -> participacao do Brasil
  - preco mundial do acucar (World Bank) -> elo de materia-prima (cana)
  - preco do petroleo Brent (World Bank) -> elo de demanda (gasolina)
  - CPI-U dos EUA -> preco real

Saida: data/dataset_anual.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    preco = pd.read_csv(DATA / "ethanol_price_oecdfao.csv")
    preco = preco[preco["is_projection"] == 0][["year", "eth_world_ref_usd_hl", "eth_usa_usd_hl"]]

    prod = pd.read_csv(DATA / "ethanol_production_eia_tbpd.csv")
    acucar = pd.read_csv(DATA / "sugar_price_world_annual.csv")  # year, sugar_world_usd_kg
    brent = pd.read_csv(DATA / "brent_oil_annual.csv")
    cpi = pd.read_csv(DATA / "us_cpi_annual.csv")

    df = (
        preco.merge(prod, on="year", how="inner")
        .merge(acucar, on="year", how="left")
        .merge(brent, on="year", how="left")
        .merge(cpi, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )

    cpi_base = float(cpi.loc[cpi["year"] == 2022, "cpi"].iloc[0])
    df["preco_etanol_real_usd_hl"] = df["eth_world_ref_usd_hl"] * (cpi_base / df["cpi"])
    df.rename(columns={"eth_world_ref_usd_hl": "preco_etanol_nom_usd_hl"}, inplace=True)

    df["share_brasil"] = df["prod_brasil_tbpd"] / df["prod_mundo_tbpd"]

    # Logs
    df["lp_nom"] = np.log(df["preco_etanol_nom_usd_hl"])
    df["lp_real"] = np.log(df["preco_etanol_real_usd_hl"])
    df["lq"] = np.log(df["prod_brasil_tbpd"])          # producao BR de etanol
    df["lp_acucar"] = np.log(df["sugar_world_usd_kg"])  # preco mundial do acucar
    df["lp_oil"] = np.log(df["brent_usd_bbl"])          # petroleo Brent

    cols = ["year", "preco_etanol_nom_usd_hl", "preco_etanol_real_usd_hl",
            "eth_usa_usd_hl", "prod_brasil_tbpd", "prod_eua_tbpd", "prod_mundo_tbpd",
            "share_brasil", "sugar_world_usd_kg", "brent_usd_bbl", "cpi",
            "lp_nom", "lp_real", "lq", "lp_acucar", "lp_oil"]
    df = df[cols]
    df.to_csv(DATA / "dataset_anual.csv", index=False)

    print(f"Periodo: {int(df.year.min())}-{int(df.year.max())}  (n={len(df)})")
    print(df[["year", "preco_etanol_real_usd_hl", "prod_brasil_tbpd", "share_brasil"]].round(3).tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
