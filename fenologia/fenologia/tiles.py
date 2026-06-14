"""
Matemática dos tiles senoidais VIIRS/MODIS (h, v) e definição do grid alvo por
tile, alinhado a um *lattice global* (limites de pixel em múltiplos de
``TARGET_RES_DEG``).

Esse alinhamento global é o que permite que os COGs cacheados por tile (VIIRS e
MapBiomas) e os templates por hexágono se encaixem por simples recorte/janela,
sem reprojeções redundantes.
"""
from __future__ import annotations

import math
import re

import numpy as np
import xarray as xr
from pyproj import Transformer

from . import config
from .rasterutils import make_target_grid

# Constantes da grade senoidal global (VIIRS/MODIS).
_R = 6371007.181
_X_MIN = -20015109.354  # x do canto oeste do tile h00
_Y_MAX = 10007554.677   # y do canto norte do tile v00
_TILE = 2 * abs(_X_MIN) / 36.0  # 1111950.5197 m (10°)

_N_H = 36
_N_V = 18

_TILE_RE = re.compile(r"h(\d{2})v(\d{2})")

_TO_LL = Transformer.from_crs(config.VIIRS_SINU_PROJ4, "EPSG:4326", always_xy=True)
_TO_SINU = Transformer.from_crs("EPSG:4326", config.VIIRS_SINU_PROJ4, always_xy=True)


def parse_tile(text: str) -> tuple[int, int]:
    """Extrai (h, v) de um GranuleUR / nome de arquivo (ex.: ...h12v10...)."""
    m = _TILE_RE.search(text)
    if not m:
        raise ValueError(f"Não encontrei hXXvYY em {text!r}")
    return int(m.group(1)), int(m.group(2))


def tile_sinu_bounds(h: int, v: int) -> tuple[float, float, float, float]:
    """Bounds senoidais (ulx, uly, lrx, lry) do tile (h, v)."""
    ulx = _X_MIN + h * _TILE
    uly = _Y_MAX - v * _TILE
    return ulx, uly, ulx + _TILE, uly - _TILE


def tile_latlon_bbox(h: int, v: int) -> tuple[float, float, float, float] | None:
    """
    Bounding box (minx, miny, maxx, maxy) do tile em EPSG:4326.

    Amostra pontos na grade do tile, transforma para lon/lat e pega os
    extremos finitos. Retorna None se o tile cair totalmente fora do globo.
    """
    ulx, uly, lrx, lry = tile_sinu_bounds(h, v)
    xs = np.linspace(ulx, lrx, 25)
    ys = np.linspace(uly, lry, 25)
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = _TO_LL.transform(gx.ravel(), gy.ravel())
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    ok = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    if not ok.any():
        return None
    return float(lon[ok].min()), float(lat[ok].min()), float(lon[ok].max()), float(lat[ok].max())


def bucket_grid(
    tiles_hv: list[tuple[int, int]], res: float = config.TARGET_RES_DEG
) -> xr.DataArray:
    """
    Template (EPSG:4326, alinhado ao lattice global) cobrindo a união dos
    bboxes lon/lat dos tiles do "balde" ``tiles_hv``.

    É o grid em memória compartilhado por todos os hexágonos cujo conjunto de
    tiles VIIRS é exatamente ``tiles_hv`` — o EVI e o MapBiomas são lidos uma
    vez nesse grid e reaproveitados (em memória) por todos eles.
    """
    bboxes = [tile_latlon_bbox(h, v) for h, v in tiles_hv]
    bboxes = [b for b in bboxes if b is not None]
    if not bboxes:
        raise ValueError(f"Nenhum tile válido em {tiles_hv}")
    minx = min(b[0] for b in bboxes)
    miny = min(b[1] for b in bboxes)
    maxx = max(b[2] for b in bboxes)
    maxy = max(b[3] for b in bboxes)
    return make_target_grid((minx, miny, maxx, maxy), res)


def tiles_for_geometry(geom) -> list[tuple[int, int]]:
    """
    Tiles (h, v) que cobrem a geometria (Polygon em EPSG:4326).

    Converte os vértices para senoidal para achar os índices candidatos e
    confirma pela interseção do bbox lon/lat do tile com o bbox da geometria.
    """
    minx, miny, maxx, maxy = geom.bounds
    # vértices + centro
    lons = [minx, maxx, minx, maxx, (minx + maxx) / 2]
    lats = [miny, miny, maxy, maxy, (miny + maxy) / 2]
    sx, sy = _TO_SINU.transform(lons, lats)
    sx = np.asarray(sx)
    sy = np.asarray(sy)

    h0 = int(math.floor((sx.min() - _X_MIN) / _TILE))
    h1 = int(math.floor((sx.max() - _X_MIN) / _TILE))
    v0 = int(math.floor((_Y_MAX - sy.max()) / _TILE))
    v1 = int(math.floor((_Y_MAX - sy.min()) / _TILE))

    out = []
    for h in range(max(0, h0 - 1), min(_N_H, h1 + 2)):
        for v in range(max(0, v0 - 1), min(_N_V, v1 + 2)):
            bbox = tile_latlon_bbox(h, v)
            if bbox is None:
                continue
            bminx, bminy, bmaxx, bmaxy = bbox
            # interseção de bboxes
            if bminx <= maxx and bmaxx >= minx and bminy <= maxy and bmaxy >= miny:
                out.append((h, v))
    return out
