"""
Extração da série temporal de EVI (média, mínima, p25, p75) por classe.

Para cada classe presente no raster de classificação (já reamostrado ao grid do
VIIRS), mascara o cubo de EVI e agrega NO ESPAÇO (x, y) para CADA data de
composição nativa do VIIRS (não há nenhuma agregação temporal: as datas são as
mesmas do dado original).

- ``extract_class_series`` produz o resultado *long* (uma linha por classe/data).
- ``to_wide`` pivota para o formato *wide* pedido: uma linha por
  ``(id_hexagono, data)`` e colunas ``evi_medio_<classe>``, ``evi_min_<classe>``,
  ``evi_p25_<classe>``, ``evi_p75_<classe>``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from . import config

# nome interno (long) -> prefixo da coluna (wide)
_STAT_RENAME = {
    "evi_media": "evi_medio",
    "evi_min": "evi_min",
    "evi_p25": "evi_p25",
    "evi_p75": "evi_p75",
}


def _pixel_area_ha(template_lat: float, res_deg: float) -> float:
    """Área aproximada de um pixel (graus) em hectares, na latitude dada."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(template_lat))
    return (res_deg * m_per_deg_lat) * (res_deg * m_per_deg_lon) / 10_000.0


def extract_class_series(
    evi_cube: xr.DataArray,
    classes_da: xr.DataArray,
    class_map: dict[int, str],
    *,
    hex_id: str,
    year: int,
    source: str,
    res_deg: float,
) -> pd.DataFrame:
    """
    Parâmetros
    ----------
    evi_cube : DataArray (time, y, x) com EVI físico (NaN onde inválido).
    classes_da : DataArray (y, x) com os códigos de classe alinhados ao cubo.
    class_map : {codigo: nome} das classes a extrair.
    hex_id, year, source : metadados anexados a cada linha.
    res_deg : resolução do grid (para estimar área).

    Retorna
    -------
    DataFrame com colunas:
      id_hexagono, ano, fonte, classe_codigo, classe_nome, data_composicao,
      evi_media, evi_min, evi_p25, evi_p75, n_pixels, n_validos, area_ha
    """
    rows = []

    # latitude central para estimativa de área
    lat_c = float(np.asarray(evi_cube["y"]).mean())
    px_area_ha = _pixel_area_ha(lat_c, res_deg)

    classes_present = set(np.unique(np.asarray(classes_da)).tolist())

    for code, name in class_map.items():
        if code not in classes_present:
            continue

        mask = classes_da == code
        n_pixels = int(mask.sum().item())
        if n_pixels == 0:
            continue

        sub = evi_cube.where(mask)  # (time, y, x), NaN fora da classe

        media = sub.mean(dim=("y", "x"), skipna=True)
        vmin = sub.min(dim=("y", "x"), skipna=True)
        p25 = sub.quantile(0.25, dim=("y", "x"), skipna=True)
        p75 = sub.quantile(0.75, dim=("y", "x"), skipna=True)
        n_validos = sub.notnull().sum(dim=("y", "x"))

        df = pd.DataFrame(
            {
                "data_composicao": pd.to_datetime(np.asarray(evi_cube["time"])),
                "evi_media": np.asarray(media),
                "evi_min": np.asarray(vmin),
                "evi_p25": np.asarray(p25),
                "evi_p75": np.asarray(p75),
                "n_validos": np.asarray(n_validos).astype("int32"),
            }
        )
        df.insert(0, "id_hexagono", hex_id)
        df.insert(1, "ano", year)
        df.insert(2, "fonte", source)
        df.insert(3, "classe_codigo", code)
        df.insert(4, "classe_nome", name)
        df["n_pixels"] = n_pixels
        df["area_ha"] = round(n_pixels * px_area_ha, 4)
        rows.append(df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "id_hexagono", "ano", "fonte", "classe_codigo", "classe_nome",
                "data_composicao", "evi_media", "evi_min", "evi_p25", "evi_p75",
                "n_validos", "n_pixels", "area_ha",
            ]
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Formato WIDE (uma linha por id_hexagono x data; colunas por estatística/classe)
# ---------------------------------------------------------------------------
def canonical_class_names() -> list[str]:
    """Ordem canônica das classes (cobertura primeiro, depois safrinha)."""
    names = list(config.COVERAGE_AGRI_CLASSES.values()) + list(
        config.SECOND_CROP_CLASSES.values()
    )
    # preserva ordem removendo eventuais duplicatas de nome
    return list(dict.fromkeys(names))


def full_wide_columns() -> list[str]:
    """Lista completa de colunas do formato wide (schema fixo para todo o Brasil)."""
    cols = ["id_hexagono", "data"]
    for name in canonical_class_names():
        for stat in ("evi_medio", "evi_min", "evi_p25", "evi_p75"):
            cols.append(f"{stat}_{name}")
    return cols


def to_wide(df_long: pd.DataFrame, ensure_full_schema: bool = True) -> pd.DataFrame:
    """
    Pivota o DataFrame *long* para o formato *wide* pedido:

        id_hexagono | data | evi_medio_<classe> | evi_min_<classe> |
        evi_p25_<classe> | evi_p75_<classe> | ...

    Cada linha é uma data de composição nativa do VIIRS. Os nomes de classe são
    globalmente únicos (as classes de safrinha têm prefixo ``segunda_safra``),
    então não há colisão entre as fontes ``coverage`` e ``second_crop``.

    Se ``ensure_full_schema`` (padrão), todas as colunas possíveis (de
    ``config``) são garantidas — classes ausentes no hexágono ficam NaN. Isso
    mantém o schema idêntico entre todos os hexágonos.
    """
    full_cols = full_wide_columns()
    if df_long is None or df_long.empty:
        return pd.DataFrame(columns=full_cols)

    df = df_long.rename(columns={"data_composicao": "data"})
    wide = df.pivot_table(
        index=["id_hexagono", "data"],
        columns="classe_nome",
        values=["evi_media", "evi_min", "evi_p25", "evi_p75"],
        aggfunc="first",
    )
    # achata o MultiIndex de colunas (stat, classe) -> "{stat}_{classe}"
    wide.columns = [f"{_STAT_RENAME[stat]}_{classe}" for stat, classe in wide.columns]
    wide = wide.reset_index()

    if ensure_full_schema:
        wide = wide.reindex(columns=full_cols)
    else:
        wide = wide[[c for c in full_cols if c in wide.columns]]

    return wide.sort_values(["id_hexagono", "data"]).reset_index(drop=True)
