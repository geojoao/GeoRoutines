"""
MapBiomas como máscara anual do VIIRS, com **pré-download** para o disco local
do pod e resolução de ano com *clamp* (fallback para o último/primeiro ano
disponível).

Duas fontes:
- ``coverage`` (macroclasse Agricultura): Collection 10 pública, com URL limpa
  por ano (existe desde 1985 até o último ano lançado) —
  ``.../collection_10/lulc/coverage/brazil_coverage_{year}.tif``.
- ``second_crop`` (safrinha): bucket de *downloads* por UUID, disponível apenas
  para 2020–2024. Para anos fora desse intervalo, usa o ano disponível mais
  próximo (*clamp*).

Regra de ano (pedido do produto): a máscara é anual; se o VIIRS estiver num ano
posterior ao último MapBiomas lançado, mantém-se o último GeoTIFF como máscara.
Aqui isso é generalizado para um *clamp* ao intervalo disponível de cada fonte.

A reamostragem de ~30 m -> ~463 m usa ``Resampling.mode`` (maioria).
``read_mapbiomas_on_grid`` lê a janela do ``template`` via ``WarpedVRT`` (sem
copiar o raster inteiro): dos arquivos locais baixados por ``download_masks``,
ou remotamente via ``/vsicurl/`` como fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

import rasterio
import requests
import xarray as xr
from rasterio.enums import Resampling
from rasterio.env import Env
from rasterio.vrt import WarpedVRT

from . import config

# ---------------------------------------------------------------------------
# URLs das fontes
# ---------------------------------------------------------------------------
_COVERAGE_URL = (
    "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
    "collection_10/lulc/coverage/brazil_coverage_{year}.tif"
)

_SECOND_CROP_BASE = "https://storage.googleapis.com/mapbiomas-downloads/public/brazil/maps"
_SECOND_CROP_UUIDS = {
    2020: "6b693804-82ed-4e1e-98ce-472b41a8d0b4",
    2021: "1a0f0412-2bb2-4a8f-b87a-8333e58ff076",
    2022: "138bb332-92ab-4e72-9ea0-1b915472fe5e",
    2023: "ce1ce9ae-9432-454a-ae29-6bf44860176c",
    2024: "f968dc13-8eb5-4c84-a28f-d603591dd6b6",
}
_SECOND_CROP_YEARS = sorted(_SECOND_CROP_UUIDS)

# Primeiro ano do coverage na Collection 10 (existe desde 1985).
_COVERAGE_MIN_YEAR = 1985


def _coverage_url(year: int) -> str:
    return _COVERAGE_URL.format(year=year)


def _second_crop_url(year: int) -> str:
    uuid = _SECOND_CROP_UUIDS[year]
    filename = f"{year}_agriculture_agricultural_use_second_crop_1-1-1_{uuid}.tif"
    return f"{_SECOND_CROP_BASE}/{uuid}/{filename}"


def _remote_url(year: int, kind: str) -> str:
    return _coverage_url(year) if kind == "coverage" else _second_crop_url(year)


# ---------------------------------------------------------------------------
# Detecção do último ano de coverage disponível (clamp superior)
# ---------------------------------------------------------------------------
def _url_ok(url: str) -> bool:
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30, stream=True)
        ok = r.status_code in (200, 206)
        r.close()
        return ok
    except Exception:
        return False


_latest_coverage_year: int | None = None


def latest_coverage_year() -> int:
    """Maior ano com coverage publicado (probe descendente a partir de hoje)."""
    global _latest_coverage_year
    if _latest_coverage_year is not None:
        return _latest_coverage_year
    top = config.end_date().year + 1
    for year in range(top, _COVERAGE_MIN_YEAR - 1, -1):
        if _url_ok(_coverage_url(year)):
            _latest_coverage_year = year
            return year
    raise RuntimeError("Nenhum coverage do MapBiomas encontrado no GCS.")


def effective_year(viirs_year: int, kind: str) -> int:
    """
    Ano do GeoTIFF MapBiomas a usar como máscara para ``viirs_year``.

    coverage    -> min(viirs_year, último ano lançado) (existe desde 1985);
    second_crop -> clamp ao intervalo disponível (2020–2024).
    """
    if kind == "coverage":
        return min(max(viirs_year, _COVERAGE_MIN_YEAR), latest_coverage_year())
    # second_crop
    if viirs_year < _SECOND_CROP_YEARS[0]:
        return _SECOND_CROP_YEARS[0]
    if viirs_year > _SECOND_CROP_YEARS[-1]:
        return _SECOND_CROP_YEARS[-1]
    return viirs_year


# ---------------------------------------------------------------------------
# Caminho local / download
# ---------------------------------------------------------------------------
def _local_path(eff_year: int, kind: str) -> Path:
    return Path(config.MAPBIOMAS_DIR) / f"{kind}_{eff_year}.tif"


def required_effective_files(viirs_years) -> set[tuple[int, str]]:
    """Conjunto de (ano_efetivo, kind) necessários para os anos VIIRS dados."""
    out: set[tuple[int, str]] = set()
    for y in set(viirs_years):
        for kind in ("coverage", "second_crop"):
            out.add((effective_year(y, kind), kind))
    return out


def _download(url: str, dest: Path) -> None:
    """Download resumível (Range) com skip se o tamanho local já bate."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length", 0))

    resume_from, mode = 0, "wb"
    if dest.exists():
        local = dest.stat().st_size
        if remote_size and local == remote_size:
            print(f"[mapbiomas] {dest.name} já baixado ({local} bytes)")
            return
        if remote_size and local < remote_size:
            resume_from, mode = local, "ab"

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
    print(f"[mapbiomas] baixado {dest.name} ({dest.stat().st_size} bytes)")


def download_masks(viirs_years) -> None:
    """
    Baixa para ``MAPBIOMAS_DIR`` **todas** as máscaras (coverage + second_crop,
    já resolvidas por ``effective_year``) necessárias para os anos VIIRS que
    serão processados neste run. Idempotente (pula o que já existe).
    """
    needed = sorted(required_effective_files(viirs_years))
    print(f"[mapbiomas] máscaras necessárias neste run: "
          f"{[(k, y) for (y, k) in needed]}")
    for eff_year, kind in needed:
        dest = _local_path(eff_year, kind)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        _download(_remote_url(eff_year, kind), dest)


def mapbiomas_path(viirs_year: int, kind: str) -> str:
    """
    Caminho do GeoTIFF para (ano VIIRS, tipo), resolvendo o ano efetivo.

    Prefere o arquivo local baixado por ``download_masks``; se não existir,
    devolve um ``/vsicurl/`` para leitura remota via GDAL (range requests).
    """
    eff_year = effective_year(viirs_year, kind)
    local = _local_path(eff_year, kind)
    if local.exists() and local.stat().st_size > 0:
        return str(local)
    return f"/vsicurl/{_remote_url(eff_year, kind)}"


def read_mapbiomas_on_grid(viirs_year: int, kind: str, template: xr.DataArray) -> xr.DataArray:
    """
    Lê o MapBiomas (ano efetivo de ``viirs_year``, ``kind``), reamostrado por
    maioria (``Resampling.mode``) direto para o grid ``template`` — sem cache
    em disco além do arquivo-fonte. Retorna DataArray (y, x) int16.
    """
    transform = template.rio.transform()
    width = template.sizes["x"]
    height = template.sizes["y"]

    src_path = mapbiomas_path(viirs_year, kind)
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
        name=f"mapbiomas_{kind}_{viirs_year}",
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)
    da = da.rio.write_nodata(config.MAPBIOMAS_NODATA)
    return da
