"""
Orquestração incremental por "balde de tiles".

Diferenças em relação à rotina ``fenologia`` (batch por ano fixo):
- a janela temporal é ``[VIIRS_START_DATE, hoje]`` (ou o env de fim);
- para cada balde, decide o que falta processar a partir da **última data já
  gravada** por hexágono (``state.load_last_date_by_hexagon``): hexágono novo →
  série inteira; hexágono existente → só as datas posteriores;
- processa **ano a ano** dentro da janela (constrói o cubo do balde de um ano
  por vez), mantendo o teto de memória em ~1 ano de cubo (evita OOM);
- a máscara MapBiomas é anual, com *clamp* ao ano disponível
  (``mapbiomas.effective_year``).
"""
from __future__ import annotations

import ctypes
import datetime as dt
import gc
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from shapely.geometry import mapping

from . import config, extract, mapbiomas, tiles, viirs
from .grid import cell_to_polygon

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _release_memory():
    gc.collect()
    if _libc is not None:
        _libc.malloc_trim(0)


def _clip_to_hex(da, geom):
    return da.rio.clip([mapping(geom)], crs="EPSG:4326", drop=False, all_touched=True)


def _hex_window(hex_id: str) -> tuple[float, float, float, float]:
    geom = cell_to_polygon(hex_id)
    minx, miny, maxx, maxy = geom.bounds
    b = config.BBOX_BUFFER_DEG
    return minx - b, miny - b, maxx + b, maxy + b


def _needed_start(hex_id: str, last_date_by_hex: dict[str, dt.date],
                  global_start: dt.date) -> dt.date:
    """Primeira data que este hexágono ainda precisa (exclusiva da última gravada)."""
    ld = last_date_by_hex.get(hex_id)
    if ld is None:
        return global_start
    return max(global_start, ld + dt.timedelta(days=1))


def process_tile_bucket_incremental(
    hex_ids: Sequence[str],
    tiles_hv: Sequence[tuple[int, int]],
    last_date_by_hex: dict[str, dt.date],
    global_start: dt.date,
    global_end: dt.date,
    res_deg: float = config.TARGET_RES_DEG,
) -> dict[str, pd.DataFrame]:
    """
    Processa incrementalmente todos os ``hex_ids`` do balde ``tiles_hv``.

    Retorna ``{hex_id: DataFrame wide}`` **apenas com as linhas novas** (datas
    ainda não gravadas). Hexágonos sem nada novo (ou sem agricultura) não
    aparecem no resultado.
    """
    template = tiles.bucket_grid(list(tiles_hv), res_deg)
    label = tiles.bucket_label(list(tiles_hv))

    # Janela de download do balde = menor "needed_start" entre seus hexágonos.
    win_start = min(_needed_start(h, last_date_by_hex, global_start) for h in hex_ids)
    if win_start > global_end:
        return {}

    # Há alguma composição nova no CMR nessa janela? (busca leve, sem download)
    latest = viirs.latest_composite_date(
        list(tiles_hv), template, win_start.isoformat(), global_end.isoformat()
    )
    if latest is None:
        return {}

    frames_by_hex: dict[str, list[pd.DataFrame]] = defaultdict(list)
    geoms = {h: cell_to_polygon(h) for h in hex_ids}
    windows = {h: _hex_window(h) for h in hex_ids}
    ld64 = {h: (np.datetime64(last_date_by_hex[h]) if h in last_date_by_hex else None)
            for h in hex_ids}

    for year in range(win_start.year, global_end.year + 1):
        y_start = max(win_start, dt.date(year, 1, 1))
        y_end = min(global_end, dt.date(year, 12, 31))
        if y_start > y_end:
            continue

        with tempfile.TemporaryDirectory(prefix="viirs_evi_") as tmp:
            cube = viirs.build_tile_cube_range(
                list(tiles_hv), y_start.isoformat(), y_end.isoformat(),
                template, Path(tmp), f"{label} {year}",
            )
        if cube is None:
            continue
        # Amarra cada data ao seu ano-calendário (a busca pode trazer a virada).
        cube = cube.sel(time=cube["time"].dt.year == year)
        if cube.sizes["time"] == 0:
            continue

        cov = mapbiomas.read_mapbiomas_on_grid(year, "coverage", template)
        sec = mapbiomas.read_mapbiomas_on_grid(year, "second_crop", template)

        for hex_id in hex_ids:
            # só datas ainda não gravadas para este hexágono
            chex = cube
            if ld64[hex_id] is not None:
                chex = cube.sel(time=cube["time"] > ld64[hex_id])
            if chex.sizes["time"] == 0:
                continue

            window = windows[hex_id]
            geom = geoms[hex_id]
            cube_hex = chex.rio.clip_box(*window)
            cov_hex = _clip_to_hex(cov.rio.clip_box(*window), geom)
            sec_hex = _clip_to_hex(sec.rio.clip_box(*window), geom)

            df_cov = extract.extract_class_series(
                cube_hex, cov_hex, config.COVERAGE_AGRI_CLASSES,
                hex_id=hex_id, year=year, source="coverage", res_deg=res_deg,
            )
            df_sec = extract.extract_class_series(
                cube_hex, sec_hex, config.SECOND_CROP_CLASSES,
                hex_id=hex_id, year=year, source="second_crop", res_deg=res_deg,
            )
            if len(df_cov):
                frames_by_hex[hex_id].append(df_cov)
            if len(df_sec):
                frames_by_hex[hex_id].append(df_sec)

        del cube, cov, sec
        _release_memory()

    result: dict[str, pd.DataFrame] = {}
    for hex_id, frames in frames_by_hex.items():
        if frames:
            result[hex_id] = extract.to_wide(pd.concat(frames, ignore_index=True))
    return result


def buckets_for_hexagons(hex_ids: Sequence[str]) -> dict[tuple, list[str]]:
    """Agrupa hexágonos por conjunto de tiles VIIRS (o "balde")."""
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for hex_id in hex_ids:
        tiles_hv = tuple(sorted(tiles.tiles_for_geometry(cell_to_polygon(hex_id))))
        buckets[tiles_hv].append(hex_id)
    return buckets


def processing_years(last_date_by_hex, hex_ids, global_start, global_end) -> list[int]:
    """Anos VIIRS que serão tocados neste run (para pré-baixar as máscaras)."""
    starts = [_needed_start(h, last_date_by_hex, global_start) for h in hex_ids]
    if not starts:
        return []
    y0 = min(starts).year
    return list(range(y0, global_end.year + 1))
