"""
Utilidades de raster: construção do grid alvo regular (EPSG:4326) para onde o
VIIRS é reprojetado e o MapBiomas reamostrado.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import xarray as xr
from affine import Affine

from . import config


def make_target_grid(bounds: Sequence[float], res: float = config.TARGET_RES_DEG) -> xr.DataArray:
    """
    Cria um DataArray vazio (template) em EPSG:4326 que define o grid alvo.

    Os limites são "snapados" a um *lattice global* (bordas de pixel em
    múltiplos de ``res``). Assim todos os templates (por tile e por hexágono)
    compartilham a mesma malha global e podem ser recortados/combinados sem
    deslocamento de meio-pixel — base de toda a estratégia de cache.

    Parâmetros
    ----------
    bounds : (minx, miny, maxx, maxy)
    res : tamanho do pixel em graus.
    """
    minx, miny, maxx, maxy = bounds
    # snap ao lattice global (múltiplos de res)
    minx = math.floor(minx / res) * res
    maxx = math.ceil(maxx / res) * res
    miny = math.floor(miny / res) * res
    maxy = math.ceil(maxy / res) * res

    width = max(1, int(round((maxx - minx) / res)))
    height = max(1, int(round((maxy - miny) / res)))

    transform = Affine.translation(minx, maxy) * Affine.scale(res, -res)

    xs = minx + (np.arange(width) + 0.5) * res
    ys = maxy - (np.arange(height) + 0.5) * res

    da = xr.DataArray(
        np.zeros((height, width), dtype="float32"),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="template",
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)
    return da
