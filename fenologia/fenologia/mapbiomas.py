"""
Leitura e reamostragem do MapBiomas para o grid do VIIRS.

Cada ano possui dois GeoTIFFs (EPSG:4326, ~30 m, uint8, nodata=0), já
salvos como COGs com overviews:
- ``{year}_coverage_lclu*.tif``: cobertura/uso geral (legenda completa).
- ``{year}_agriculture_agricultural_use_second_crop*.tif``: classe da 2ª safra.

A reamostragem de 30 m -> ~463 m usa ``Resampling.mode`` (maioria), equivalente
ao ``salem.lookup_transform(method=moda)`` do script de referência.

Como o COG já vem tilado/com overviews, ``read_mapbiomas_on_grid`` lê
diretamente (via ``WarpedVRT``) a janela do grid alvo (``template``) — sem
nenhum cache em disco: GDAL busca só os blocos do GeoTIFF que cobrem a janela.
"""
from __future__ import annotations

import glob
from pathlib import Path

import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from . import config


def mapbiomas_path(year: int, kind: str) -> Path:
    """Resolve o caminho do GeoTIFF do MapBiomas para (ano, tipo)."""
    pattern = config.MAPBIOMAS_PATTERNS[kind].format(year=year)
    matches = sorted(glob.glob(str(Path(config.MAPBIOMAS_DIR) / pattern)))
    if not matches:
        raise FileNotFoundError(
            f"GeoTIFF do MapBiomas não encontrado: ano={year}, tipo={kind}, "
            f"padrão={pattern} em {config.MAPBIOMAS_DIR}"
        )
    return Path(matches[0])


def read_mapbiomas_on_grid(year: int, kind: str, template: xr.DataArray) -> xr.DataArray:
    """
    Lê o MapBiomas (ano, tipo), reamostrado por maioria (``Resampling.mode``)
    direto para o grid ``template`` (EPSG:4326), via ``WarpedVRT`` — sem cache
    em disco.

    Retorna um DataArray (y, x) inteiro (int16) alinhado ao template.
    """
    transform = template.rio.transform()
    width = template.sizes["x"]
    height = template.sizes["y"]

    src_path = mapbiomas_path(year, kind)
    with rasterio.open(src_path) as src:
        vrt_opts = dict(
            crs="EPSG:4326",
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.mode,
            nodata=config.MAPBIOMAS_NODATA,
        )
        with WarpedVRT(src, **vrt_opts) as vrt:
            arr = vrt.read(1)

    da = xr.DataArray(
        arr.astype("int16"),
        coords={"y": template["y"], "x": template["x"]},
        dims=("y", "x"),
        name=f"mapbiomas_{kind}_{year}",
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)
    da = da.rio.write_nodata(config.MAPBIOMAS_NODATA)
    return da
