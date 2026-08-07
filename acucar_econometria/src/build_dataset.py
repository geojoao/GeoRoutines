"""
Constroi o painel anual usado no estudo econometrico.

Junta:
  - preco mundial do acucar (World Bank Pink Sheet, "Sugar, world", US$/kg nominal)
  - producao brasileira de acucar bruto centrifugado e de cana (FAOSTAT)
  - producao mundial de acucar bruto centrifugado (FAOSTAT) -> participacao do Brasil
  - CPI-U dos EUA (BLS, media anual, 1982-84=100) -> preco real

Saida: data/dataset_anual.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    preco = pd.read_csv(DATA / "sugar_price_world_annual.csv")            # year, sugar_world_usd_kg
    prod_br = pd.read_csv(DATA / "brazil_sugar_production_fao.csv")       # year, raw_sugar_t, sugarcane_t
    prod_w = pd.read_csv(DATA / "world_sugar_production_fao.csv")         # year, world_raw_sugar_t
    cpi = pd.read_csv(DATA / "us_cpi_annual.csv")                         # year, cpi

    df = (
        preco.merge(prod_br, on="year", how="inner")
        .merge(prod_w, on="year", how="left")
        .merge(cpi, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )

    # Preco real (US$/kg de 2023) usando o CPI-U
    cpi_base = float(cpi.loc[cpi["year"] == 2023, "cpi"].iloc[0])
    df["preco_real_usd_kg"] = df["sugar_world_usd_kg"] * (cpi_base / df["cpi"])
    df.rename(columns={"sugar_world_usd_kg": "preco_nom_usd_kg"}, inplace=True)

    # Participacao do Brasil na producao mundial de acucar
    df["share_brasil"] = df["raw_sugar_t"] / df["world_raw_sugar_t"]

    # Logs (base natural)
    df["lp_nom"] = np.log(df["preco_nom_usd_kg"])
    df["lp_real"] = np.log(df["preco_real_usd_kg"])
    df["lq_acucar"] = np.log(df["raw_sugar_t"])       # producao brasileira de acucar
    df["lq_cana"] = np.log(df["sugarcane_t"])         # producao brasileira de cana (robustez)

    cols = [
        "year", "preco_nom_usd_kg", "preco_real_usd_kg", "cpi",
        "raw_sugar_t", "sugarcane_t", "world_raw_sugar_t", "share_brasil",
        "lp_nom", "lp_real", "lq_acucar", "lq_cana",
    ]
    df = df[cols]
    out = DATA / "dataset_anual.csv"
    df.to_csv(out, index=False)

    print(f"Dataset salvo em {out}")
    print(f"Periodo: {int(df.year.min())}-{int(df.year.max())}  (n={len(df)})")
    print(df[["year", "preco_real_usd_kg", "raw_sugar_t", "share_brasil"]].tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
