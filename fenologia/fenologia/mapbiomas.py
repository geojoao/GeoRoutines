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

Quando os arquivos locais não existem, o módulo abre os COGs diretamente do
GCS via ``/vsicurl/`` (GDAL virtual filesystem), realizando apenas os range
requests necessários para a janela do template — sem download completo.
"""
from __future__ import annotations

import glob
from pathlib import Path

import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from . import config

# URLs públicos dos COGs no GCS (MapBiomas Brasil — Collection 9)
_GCS_BASE = "https://storage.googleapis.com/mapbiomas-downloads/public/brazil/maps"
_GCS_UUIDS = {
    "coverage": {
        2020: "ac1d9d62-6d91-4564-bc89-452c369af273",
        2021: "a4abb8ef-433d-45c0-9e1b-56141b9a83b7",
        2022: "2361a4a7-8905-4672-b3b9-498a52407de7",
        2023: "2242342e-4c61-44e4-83ac-2a2e7e23b148",
        2024: "e04830ad-d1b1-4739-b27b-ffea882f0d77",
    },
    "second_crop": {
        2020: "6b693804-82ed-4e1e-98ce-472b41a8d0b4",
        2021: "1a0f0412-2bb2-4a8f-b87a-8333e58ff076",
        2022: "138bb332-92ab-4e72-9ea0-1b915472fe5e",
        2023: "ce1ce9ae-9432-454a-ae29-6bf44860176c",
        2024: "f968dc13-8eb5-4c84-a28f-d603591dd6b6",
    },
}
_GCS_TYPE_NAMES = {
    "coverage": "coverage_lclu",
    "second_crop": "agriculture_agricultural_use_second_crop",
}


def _gcs_vsicurl_path(year: int, kind: str) -> str:
    """Retorna o caminho /vsicurl/ para o COG no GCS."""
    uuid = _GCS_UUIDS[kind][year]
    type_name = _GCS_TYPE_NAMES[kind]
    filename = f"{year}_{type_name}_1-1-1_{uuid}.tif"
    return f"/vsicurl/{_GCS_BASE}/{uuid}/{filename}"


def mapbiomas_path(year: int, kind: str) -> str:
    """
    Resolve o caminho do GeoTIFF do MapBiomas para (ano, tipo).

    Tenta primeiro o arquivo local em MAPBIOMAS_DIR; se não encontrar,
    retorna um caminho /vsicurl/ para o COG no GCS (leitura remota via
    GDAL range requests — funciona sem download completo pois são COGs).
    """
    pattern = config.MAPBIOMAS_PATTERNS[kind].format(year=year)
    local_dir = Path(config.MAPBIOMAS_DIR)
    matches = sorted(glob.glob(str(local_dir / pattern)))
    if matches:
        return matches[0]
    return _gcs_vsicurl_path(year, kind)


def read_mapbiomas_on_grid(year: int, kind: str, template: xr.DataArray) -> xr.DataArray:
    """
    Lê o MapBiomas (ano, tipo), reamostrado por maioria (``Resampling.mode``)
    direto para o grid ``template`` (EPSG:4326), via ``WarpedVRT`` — sem cache
    em disco.

    Retorna um DataArray (y, x) inteiro (int16) alinhado ao template.
    """
    from rasterio.env import Env

    transform = template.rio.transform()
    width = template.sizes["x"]
    height = template.sizes["y"]

    src_path = mapbiomas_path(year, kind)
    # GDAL_HTTP_UNSAFESSL: necessário em ambientes com certificado auto-assinado na chain
    is_remote = str(src_path).startswith("/vsicurl/")
    gdal_env = {"GDAL_HTTP_UNSAFESSL": "YES", "CPL_VSIL_CURL_USE_HEAD": "NO"} if is_remote else {}
    with Env(**gdal_env):
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
