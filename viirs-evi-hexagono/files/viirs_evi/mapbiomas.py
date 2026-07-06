"""
MapBiomas como máscara anual do VIIRS, por **leitura remota** dos COGs.

Os GeoTIFFs do MapBiomas são publicados como **COGs com overviews** no bucket
público do GCS, e a URL de cada ``(ano, produto)`` é resolvida pela **API de
export do MapBiomas** (o mesmo endpoint que a plataforma usa):

    POST https://prd.plataforma.mapbiomas.org/api/v1/brazil/maps/export
    {"year": 2015, "subthemeKey": "coverage_lclu", "territoryId": "1-1-1"}
    -> {"url": "https://storage.googleapis.com/.../2015_coverage_lclu_1-1-1_<uuid>.tif"}

Os arquivos SÃO COGs com overviews, então a leitura remota via ``/vsicurl/`` +
``WarpedVRT`` (30 m -> ~463 m) funciona — o GDAL usa o overview mais próximo da
resolução alvo. Porém a máscara é lida **uma vez por balde** e há **poucos
arquivos distintos** (um por ano): como ``nº de leituras >> nº de arquivos``,
**baixar cada COG uma vez e ler do disco local é bem mais rápido** do que
reler remotamente por balde (uma leitura remota de um tile inteiro leva dezenas
de segundos; multiplicada por ~60 baldes × N anos, inviabiliza o run completo).

Por isso o padrão é **download local** (``prefetch``): baixa os COGs
necessários ao run para ``MAPBIOMAS_DIR`` e lê janelas locais (rápido).
Para tocar poucos baldes/anos sem gastar disco, ``VIIRS_MAPBIOMAS_REMOTE=1``
força a leitura remota via ``/vsicurl`` (sem download).

Regra de ano (pedido do produto): a máscara é anual; se o VIIRS estiver num ano
**posterior** ao último MapBiomas lançado (a API responde ``status: PENDING``,
sem ``url``), usa-se o último ano disponível. Buracos pontuais caem no ano
disponível mais próximo (*clamp*).
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
# API de export do MapBiomas
# ---------------------------------------------------------------------------
_EXPORT_API = "https://prd.plataforma.mapbiomas.org/api/v1/brazil/maps/export"
_TERRITORY_BRAZIL = "1-1-1"  # país Brasil
_SUBTHEME = {
    "coverage": "coverage_lclu",
    "second_crop": "agriculture_agricultural_use_second_crop",
}
_MIN_YEAR = 1985  # primeiro ano do coverage MapBiomas Brasil

# URL pública estável do coverage (Collection 10) — todos os anos, SEM overviews.
# Usada para o coverage no modo download (padrão): baixamos o arquivo inteiro e
# lemos local, então a ausência de overviews é irrelevante e evitamos a API.
_COVERAGE_PUBLIC = (
    "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
    "collection_10/lulc/coverage/brazil_coverage_{year}.tif"
)

# Registro de UUIDs da safrinha (agriculture_agricultural_use_second_crop),
# colhidos da API de export do MapBiomas (UUIDs estáveis). Evita depender da API
# em runtime. Para uma nova coleção/ano, acrescente aqui (ou o fallback via API
# resolve automaticamente — ver ``_export_url``).
_SECOND_CROP_BASE = "https://storage.googleapis.com/mapbiomas-downloads/public/brazil/maps"
_SECOND_CROP_UUIDS = {
    2012: "20fa85e9-ae55-4364-af89-5b200fa6b36b",
    2013: "6b05d133-eea0-4545-9317-0006c5f0f823",
    2014: "b67ba13e-39b5-4e0c-9625-3b42c1e8623a",
    2015: "742e5e1a-744f-4e4a-9f10-4147398a05f3",
    2016: "e5afb7a3-7eac-4108-8cdc-4f4bde23b867",
    2017: "3447deb7-3de9-4d8e-b568-0026f8a26755",
    2018: "426286a2-798f-4bfc-a947-a76ee54c24b6",
    2019: "90c180f1-76f2-4166-b341-442b238b0d25",
    2020: "6b693804-82ed-4e1e-98ce-472b41a8d0b4",
    2021: "1a0f0412-2bb2-4a8f-b87a-8333e58ff076",
    2022: "138bb332-92ab-4e72-9ea0-1b915472fe5e",
    2023: "ce1ce9ae-9432-454a-ae29-6bf44860176c",
    2024: "f968dc13-8eb5-4c84-a28f-d603591dd6b6",
}
_SECOND_CROP_YEARS = sorted(_SECOND_CROP_UUIDS)


def _second_crop_url_from_uuid(year: int, uuid: str) -> str:
    fn = f"{year}_agriculture_agricultural_use_second_crop_1-1-1_{uuid}.tif"
    return f"{_SECOND_CROP_BASE}/{uuid}/{fn}"

# Padrão: baixar os COGs e ler localmente (mais rápido — arquivos << leituras).
# VIIRS_MAPBIOMAS_REMOTE=1 força leitura remota via /vsicurl (sem download).
_REMOTE_MODE = os.environ.get("VIIRS_MAPBIOMAS_REMOTE", "").strip().lower() in ("1", "true", "yes")
_LOCAL_MODE = not _REMOTE_MODE

# caches
_url_cache: dict[tuple[int, str], str | None] = {}
_latest_cache: dict[str, int] = {}
_resolved_cache: dict[tuple[int, str], tuple[int, str]] = {}


def _export_url(year: int, kind: str) -> str | None:
    """
    Pergunta à API a URL do COG para ``(year, kind)``.

    Retorna a ``url`` quando o ano está lançado, ou ``None`` quando a API
    responde ``status: PENDING`` (ano ainda não publicado) ou dá erro.
    """
    try:
        r = requests.post(
            _EXPORT_API,
            json={"year": year, "subthemeKey": _SUBTHEME[kind], "territoryId": _TERRITORY_BRAZIL},
            timeout=60,
        )
    except Exception:
        return None
    if r.status_code != 200:
        return None
    return r.json().get("url")  # ausente se PENDING


def _available_url(year: int, kind: str) -> str | None:
    key = (year, kind)
    if key not in _url_cache:
        _url_cache[key] = _export_url(year, kind)
    return _url_cache[key]


_GCS_COVERAGE_LISTING = (
    "https://storage.googleapis.com/storage/v1/b/mapbiomas-public/o"
    "?prefix=initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_&delimiter=/"
)


def _latest_from_listing() -> int | None:
    """Maior ano de coverage via listagem do GCS (uma chamada rápida)."""
    import re

    try:
        r = requests.get(_GCS_COVERAGE_LISTING, timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception:
        return None
    years = {int(m.group(1)) for it in items
             for m in [re.search(r"brazil_coverage_(\d{4})\.tif", it.get("name", ""))] if m}
    return max(years) if years else None


def latest_available_year(kind: str) -> int:
    """
    Maior ano publicado. Detecta pelo *listing* do GCS (rápido) e usa como teto
    para ambos os produtos (a coleção é lançada em conjunto). Cai para um probe
    descendente na API só se o listing falhar.
    """
    if kind in _latest_cache:
        return _latest_cache[kind]
    top = _latest_from_listing()
    if top is None:
        top = config.end_date().year
        for year in range(top, _MIN_YEAR - 1, -1):
            if _available_url(year, kind):
                _latest_cache[kind] = year
                return year
        raise RuntimeError(f"Nenhum MapBiomas '{kind}' disponível.")
    _latest_cache[kind] = top
    return top


def _resolve_second_crop(target: int, latest: int) -> tuple[int, str]:
    """
    Safrinha pelo registro de UUIDs (clamp ao intervalo do registro). Para anos
    além do registro (coleção nova), tenta a API como fallback e, se falhar,
    mantém o último ano do registro (regra: ano > último lançado -> último).
    """
    yrs = _SECOND_CROP_YEARS
    if target > yrs[-1]:
        url = _available_url(target, "second_crop")  # fallback API (best-effort)
        if url:
            return target, url
    y = min(max(target, yrs[0]), yrs[-1])  # clamp ao registro
    return y, _second_crop_url_from_uuid(y, _SECOND_CROP_UUIDS[y])


def resolve(year: int, kind: str) -> tuple[int, str]:
    """
    ``(ano_efetivo, url)`` para ``(year, kind)`` com *clamp* ao intervalo
    disponível (anos > último lançado -> último; buracos -> vizinho disponível).

    - **coverage**: URL pública da Collection 10 (sem API), existe 1985..último;
    - **second_crop**: resolvido pela API do MapBiomas (COG por UUID).
    """
    key = (year, kind)
    if key in _resolved_cache:
        return _resolved_cache[key]
    latest = latest_available_year(kind)
    target = min(max(year, _MIN_YEAR), latest)
    if kind == "coverage":
        out = (target, _COVERAGE_PUBLIC.format(year=target))
    else:
        out = _resolve_second_crop(target, latest)
    _resolved_cache[key] = out
    return out


def effective_year(year: int, kind: str) -> int:
    return resolve(year, kind)[0]


def required_effective_files(viirs_years) -> set[tuple[int, str]]:
    out: set[tuple[int, str]] = set()
    for y in set(viirs_years):
        for kind in ("coverage", "second_crop"):
            out.add((effective_year(y, kind), kind))
    return out


# ---------------------------------------------------------------------------
# Origem do raster (local/download por padrão; remota se VIIRS_MAPBIOMAS_REMOTE=1)
# ---------------------------------------------------------------------------
def _local_path(eff_year: int, kind: str) -> Path:
    return Path(config.MAPBIOMAS_DIR) / f"{kind}_{eff_year}.tif"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length", 0))
    resume_from, mode = 0, "wb"
    if dest.exists():
        local = dest.stat().st_size
        if remote_size and local == remote_size:
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


def prefetch(viirs_years) -> None:
    """
    Resolve (e cacheia) as URLs das máscaras necessárias neste run — falha
    cedo se algum ano não existir. No modo padrão (local), também **baixa** os
    COGs para ``MAPBIOMAS_DIR``; com ``VIIRS_MAPBIOMAS_REMOTE=1`` só resolve as
    URLs (leitura remota via ``/vsicurl``).
    """
    needed = sorted(required_effective_files(viirs_years))
    print(f"[mapbiomas] máscaras necessárias neste run: {[(k, y) for (y, k) in needed]} "
          f"| modo: {'local' if _LOCAL_MODE else 'remoto (/vsicurl)'}")
    if not _LOCAL_MODE:
        return
    for eff_year, kind in needed:
        dest = _local_path(eff_year, kind)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        _, url = resolve(eff_year, kind)
        _download(url, dest)


def mapbiomas_src(year: int, kind: str) -> str:
    """
    Caminho-fonte para (ano VIIRS, tipo): ``/vsicurl/<url>`` (remoto, padrão)
    ou o arquivo local baixado por ``prefetch`` (modo local).
    """
    eff_year, url = resolve(year, kind)
    if _LOCAL_MODE:
        local = _local_path(eff_year, kind)
        if local.exists() and local.stat().st_size > 0:
            return str(local)
    return f"/vsicurl/{url}"


def read_mapbiomas_on_grid(year: int, kind: str, template: xr.DataArray) -> xr.DataArray:
    """
    Lê o MapBiomas (ano efetivo de ``year``, ``kind``) reamostrado por maioria
    (``Resampling.mode``) direto para o grid ``template``, remotamente via
    ``WarpedVRT`` sobre ``/vsicurl/`` (aproveitando os overviews do COG).
    Retorna DataArray (y, x) int16.
    """
    transform = template.rio.transform()
    width = template.sizes["x"]
    height = template.sizes["y"]

    src_path = mapbiomas_src(year, kind)
    is_remote = str(src_path).startswith("/vsicurl/")
    gdal_env = {
        "GDAL_HTTP_UNSAFESSL": "YES",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        # cache de blocos p/ reduzir range requests repetidos entre baldes
        "CPL_VSIL_CURL_CACHE_SIZE": "200000000",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    } if is_remote else {}
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
