"""
Leitura e reamostragem do MapBiomas para o grid do VIIRS.

Cada ano possui dois GeoTIFFs (EPSG:4326, ~30 m, uint8, nodata=0):
- ``{year}_coverage_lclu*.tif``: cobertura/uso geral (legenda completa).
- ``{year}_agriculture_agricultural_use_second_crop*.tif``: classe da 2ª safra.

A reamostragem de 30 m -> ~463 m usa ``Resampling.mode`` (maioria), equivalente
ao ``salem.lookup_transform(method=moda)`` do script de referência.

**Eficiência**: a reamostragem (cara) é feita UMA vez por (tile, ano, tipo) e
salva em cache (``read_mapbiomas_tile_cog`` / ``ensure_mapbiomas_tile_cog``),
via ``WarpedVRT`` (streaming, baixa memória). Os hexágonos depois só fazem
leitura por janela do COG cacheado (``read_mapbiomas_on_grid``).
"""
from __future__ import annotations

import glob
from pathlib import Path

import rasterio
import rioxarray  # noqa: F401  (accessor .rio)
import xarray as xr
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.vrt import WarpedVRT
from rioxarray.merge import merge_arrays

from . import config, tiles


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


# ---------------------------------------------------------------------------
# Cache por (tile, ano, tipo): MapBiomas reamostrado (maioria) ao grid do tile
# ---------------------------------------------------------------------------
def mapbiomas_tile_cog_path(h: int, v: int, year: int, kind: str) -> Path:
    return (
        Path(config.TILE_CACHE_DIR)
        / "mapbiomas"
        / kind
        / str(year)
        / f"{kind}_h{h:02d}v{v:02d}_{year}.tif"
    )


def ensure_mapbiomas_tile_cog(h: int, v: int, year: int, kind: str) -> Path:
    """
    Garante o GeoTIFF do MapBiomas reamostrado (maioria) para o grid do tile
    (h, v). A reamostragem 30 m -> ~463 m é feita UMA vez por (tile, ano, tipo),
    via ``WarpedVRT`` + cópia em blocos (não carrega o tile inteiro em memória).
    """
    out = mapbiomas_tile_cog_path(h, v, year, kind)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    grid = tiles.tile_grid(h, v)
    transform = grid.rio.transform()
    width = grid.sizes["x"]
    height = grid.sizes["y"]

    src_path = mapbiomas_path(year, kind)
    tmp = out.with_suffix(".tmp.tif")
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
            rio_copy(
                vrt,
                tmp,
                driver="GTiff",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                compress="deflate",
            )
    tmp.replace(out)
    return out


def read_mapbiomas_on_grid(
    year: int,
    kind: str,
    template: xr.DataArray,
    tiles_hv: list[tuple[int, int]] | None = None,
) -> xr.DataArray:
    """
    Lê o MapBiomas (ano, tipo) já reamostrado (do cache por tile), recorta à
    janela do ``template`` e alinha ao template.

    Retorna um DataArray (y, x) inteiro (int16) alinhado ao template.
    """
    if tiles_hv is None:
        from shapely.geometry import box

        tiles_hv = tiles.tiles_for_geometry(box(*template.rio.bounds()))

    minx, miny, maxx, maxy = template.rio.bounds()
    pad = config.TARGET_RES_DEG * 2

    parts = []
    for h, v in tiles_hv:
        cog = ensure_mapbiomas_tile_cog(h, v, year, kind)
        try:
            # context manager + .load(): materializa e FECHA o dataset GDAL.
            with rioxarray.open_rasterio(cog, masked=False) as src:
                da = src.squeeze("band", drop=True) if "band" in src.dims else src
                da = da.rio.clip_box(minx - pad, miny - pad, maxx + pad, maxy + pad).load()
            parts.append(da)
        except Exception:
            continue  # tile não cobre essa janela

    if not parts:
        raise RuntimeError(f"Nenhum tile MapBiomas cobre o template ({year}, {kind}).")

    merged = parts[0] if len(parts) == 1 else merge_arrays(parts)
    resampled = merged.rio.reproject_match(template, resampling=Resampling.nearest)
    resampled = resampled.astype("int16")
    resampled = resampled.rio.write_nodata(config.MAPBIOMAS_NODATA)
    resampled.name = f"mapbiomas_{kind}_{year}"
    return resampled
