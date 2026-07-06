"""
Estado incremental derivado do ``evi_brazil.parquet`` existente.

A regra de negócio: para cada hexágono, olhamos a **última data** já presente no
parquet. Se o hexágono não existir, processa toda a série disponível; se existir,
processa só as datas posteriores à última já gravada (até a última composição
VIIRS disponível para o tile).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def load_last_date_by_hexagon(parquet_path: str | Path) -> dict[str, dt.date]:
    """
    Lê o parquet canônico e devolve ``{id_hexagono: última_data}``.

    Lê apenas as colunas ``id_hexagono`` e ``data`` (barato mesmo para o
    parquet inteiro do Brasil). Retorna ``{}`` se o arquivo não existir/vazio.
    """
    p = Path(parquet_path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    df = pd.read_parquet(p, columns=["id_hexagono", "data"])
    if df.empty:
        return {}
    s = df.groupby("id_hexagono")["data"].max()
    return {str(h): pd.Timestamp(v).date() for h, v in s.items()}
