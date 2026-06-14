"""
Acesso aos dados de EVI do VIIRS (produto VNP13A1, 500 m, composição de 16 dias)
da NASA LP DAAC, via biblioteca ``earthaccess``.

Particularidades tratadas aqui:
- Os granules vêm em HDF-EOS5 (.h5) na grade senoidal por tile (h/v). Em vez de
  depender do driver HDF5 do GDAL (que normalmente NÃO vem nos wheels do
  rasterio), lemos o array com ``h5py`` e montamos a georreferência senoidal a
  partir do ``StructMetadata.0``. Depois reprojetamos para EPSG:4326.
- Para cobrir um hexágono que cai em mais de um tile, fazemos o mosaico dos
  tiles da mesma data de composição antes de reprojetar.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import earthaccess
import h5py
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registra o accessor .rio)
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays
from tqdm import tqdm

from . import config, tiles

# Regex para extrair a data de composição (AYYYYDDD) do GranuleUR.
_DATE_RE = re.compile(r"\.A(\d{4})(\d{3})\.")
# Regex para a data no nome do COG cacheado: evi_hXXvYY_YYYYMMDD.tif
_COG_DATE_RE = re.compile(r"_(\d{8})\.tif$")


def _with_retries(fn, *args, what: str = "", attempts: int = 5, wait: float = 20.0, **kwargs):
    """Executa ``fn`` com novas tentativas em caso de erro transitório (ex.: CMR)."""
    for i in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if i == attempts:
                raise
            tqdm.write(f"    [retry {i}/{attempts}] {what}: {exc!r} (espera {wait:.0f}s)")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------
_AUTH = None


def authenticate() -> "earthaccess.Auth":
    """
    Autentica no NASA Earthdata. Estratégias tentadas em ordem:
    1. variáveis de ambiente EARTHDATA_USERNAME / EARTHDATA_PASSWORD;
    2. arquivo ~/.netrc;
    3. modo interativo (pede usuário/senha).
    """
    global _AUTH
    if _AUTH is not None and _AUTH.authenticated:
        return _AUTH
    for strategy in ("environment", "netrc", "interactive"):
        try:
            auth = earthaccess.login(strategy=strategy, persist=True)
            if auth.authenticated:
                _AUTH = auth
                return auth
        except Exception:
            continue
    raise RuntimeError(
        "Não foi possível autenticar no NASA Earthdata. Configure as variáveis "
        "EARTHDATA_USERNAME/EARTHDATA_PASSWORD ou um arquivo ~/.netrc."
    )


# ---------------------------------------------------------------------------
# Busca e download
# ---------------------------------------------------------------------------
def granule_ur(granule) -> str:
    return granule["umm"]["GranuleUR"]


def granule_date(granule) -> dt.date:
    """Data de início da composição de 16 dias (a partir do GranuleUR AYYYYDDD)."""
    m = _DATE_RE.search(granule_ur(granule))
    if not m:
        raise ValueError(f"Não consegui extrair a data de {granule_ur(granule)}")
    year, doy = int(m.group(1)), int(m.group(2))
    return dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)


def search_granules_bbox(bbox: Sequence[float], start: str, end: str) -> list:
    """
    Busca granules VNP13A1 no bounding box ``(minx, miny, maxx, maxy)`` e no
    intervalo [start, end]. A busca no CMR é anônima (não exige login).
    """
    minx, miny, maxx, maxy = bbox
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        results = earthaccess.search_data(
            short_name=config.VIIRS_SHORT_NAME,
            version=config.VIIRS_VERSION,
            bounding_box=(minx, miny, maxx, maxy),
            temporal=(start, end),
        )
    return list(results)


def search_granules(geometry, start: str, end: str) -> list:
    """Compat.: busca por bbox da geometria."""
    return search_granules_bbox(geometry.bounds, start, end)


def download_granules(granules: Sequence, cache_dir: Path | str = None) -> dict[str, Path]:
    """
    Baixa os granules para o cache local (pulando os já existentes) e retorna um
    dicionário {GranuleUR: caminho_local}.
    """
    cache_dir = Path(cache_dir or config.VIIRS_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not granules:
        return {}
    authenticate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        paths = earthaccess.download(list(granules), local_path=str(cache_dir))
    out: dict[str, Path] = {}
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        out[p.stem] = p  # stem == GranuleUR (sem .h5)
    return out


# ---------------------------------------------------------------------------
# Acesso direto ao S3 (in-region, sem egress) — para rodar perto dos dados
# ---------------------------------------------------------------------------
def use_s3_access() -> bool:
    """
    Decide se o VIIRS é acessado direto do S3 (in-region) ou baixado via HTTPS.

    - ``VIIRS_ACCESS_MODE=s3``       -> sempre S3.
    - ``VIIRS_ACCESS_MODE=download`` -> sempre HTTPS (cache .h5 local).
    - ``VIIRS_ACCESS_MODE=auto``     -> S3 quando o ambiente parecer in-region:
      ``AWS_REGION``/``AWS_DEFAULT_REGION`` == região da LP DAAC, ou o próprio
      earthaccess detectar execução in-region (EC2/SageMaker/etc.).
    """
    mode = (config.VIIRS_ACCESS_MODE or "auto").lower()
    if mode == "s3":
        return True
    if mode == "download":
        return False
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region == config.AWS_REGION:
        return True
    try:
        store = getattr(earthaccess, "__store__", None)
        if store is not None and bool(getattr(store, "in_region", False)):
            return True
    except Exception:
        pass
    return False


def open_granules(granules: Sequence):
    """
    Abre os granules direto do S3 (in-region) via ``earthaccess.open``,
    devolvendo handles fsspec na MESMA ordem da entrada. In-region o earthaccess
    devolve objetos S3 (acesso direto); fora de região, HTTPS.
    """
    if not granules:
        return []
    authenticate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        files = earthaccess.open(list(granules))
    return files


def _h5_cache_path(cache_dir: Path, granule) -> Path:
    return cache_dir / f"{granule_ur(granule)}.h5"


def _iter_s3_sources(granules: Sequence, cache_dir: Path):
    """
    Para cada granule, produz ``(granule, h5_source, cleanup)`` para gerar o COG,
    acessando o objeto direto do S3 (in-region).

    Padrão: copia o objeto do S3 para o cache local (``cache_dir``, pulando os já
    presentes — mesmo footprint do modo HTTPS) e devolve o caminho; assim
    ``read_evi_tile`` lê do disco com h5py, sem as leituras picadas que tornam o
    HDF5 lento sobre fsspec. Com ``VIIRS_S3_STREAM=1`` devolve o próprio
    file-like do S3 (sem cópia local). ``cleanup`` fecha o handle de stream.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if config.VIIRS_S3_STREAM:
        files = _with_retries(open_granules, granules, what="open S3")
        if files:
            tqdm.write(f"    [S3] streaming {len(files)} granules via {type(files[0]).__name__}")
        for g, fobj in zip(granules, files):
            yield g, fobj, fobj.close
        return

    # Copia para o cache local só os granules que ainda não estão lá.
    pending = [g for g in granules if not _h5_cache_path(cache_dir, g).exists()]
    files = _with_retries(open_granules, pending, what="open S3") if pending else []
    if files:
        tqdm.write(f"    [S3] copiando {len(files)} granules via {type(files[0]).__name__}")
    fobj_by_ur = {granule_ur(g): f for g, f in zip(pending, files)}

    for g in granules:
        local = _h5_cache_path(cache_dir, g)
        if not local.exists():
            fobj = fobj_by_ur.get(granule_ur(g))
            if fobj is None:
                continue
            tmp = local.with_suffix(".part")
            try:
                with open(tmp, "wb") as dst:
                    shutil.copyfileobj(fobj, dst, length=8 * 1024 * 1024)
                tmp.replace(local)  # escrita atômica
            finally:
                try:
                    fobj.close()
                except Exception:
                    pass
        yield g, local, lambda: None


def _safe_tile(granule) -> tuple[int, int] | None:
    try:
        return tiles.parse_tile(granule_ur(granule))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Leitura do HDF-EOS5 (.h5)
# ---------------------------------------------------------------------------
def _parse_struct_metadata(meta: str) -> dict:
    """Extrai XDim, YDim, UpperLeftPointMtrs e LowerRightMtrs do StructMetadata.0."""

    def _find(key):
        m = re.search(rf"{key}=([^\n]+)", meta)
        return m.group(1).strip() if m else None

    def _pair(key):
        raw = _find(key)
        if raw is None:
            return None
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
        return float(nums[0]), float(nums[1])

    xdim = int(re.search(r"XDim=(\d+)", meta).group(1))
    ydim = int(re.search(r"YDim=(\d+)", meta).group(1))
    ul = _pair("UpperLeftPointMtrs")
    lr = _pair("LowerRightMtrs")
    return {"xdim": xdim, "ydim": ydim, "ul": ul, "lr": lr}


def _find_evi_dataset(h5: h5py.File) -> str:
    """Localiza o caminho do Data Field do EVI (contendo 'EVI' mas não 'EVI2')."""
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1].upper()
            if "EVI" in base and "EVI2" not in base:
                found.append(name)

    h5.visititems(visitor)
    if not found:
        raise KeyError("Nenhum dataset EVI encontrado no HDF5.")
    # prefere o que contém '16 days EVI'
    found.sort(key=lambda n: ("16 DAYS EVI" not in n.upper(), len(n)))
    return found[0]


def read_evi_tile(h5_source) -> xr.DataArray:
    """
    Lê o EVI de um granule VNP13A1 e devolve um ``DataArray`` (y, x) em CRS
    senoidal, com EVI físico (float32) e fill -> NaN.

    ``h5_source`` pode ser um caminho (str/Path) OU um objeto file-like (ex.: o
    handle S3 do earthaccess) — ``h5py.File`` aceita ambos.
    """
    with h5py.File(h5_source, "r") as h5:
        ds_name = _find_evi_dataset(h5)
        ds = h5[ds_name]
        raw = ds[()].astype("float32")

        attrs = ds.attrs
        fill = float(np.array(attrs.get("_FillValue", config.VIIRS_EVI_FILL)).ravel()[0])
        scale = attrs.get("scale_factor", None)
        if scale is not None:
            scale = float(np.array(scale).ravel()[0])
            # VNP13 usa scale_factor=0.0001 (multiplicar). Se vier 10000, inverte.
            if scale > 1:
                scale = 1.0 / scale
        else:
            scale = config.VIIRS_EVI_SCALE

        vmin, vmax = config.VIIRS_EVI_VALID
        vr = attrs.get("valid_range", None)
        if vr is not None:
            vr = np.array(vr).ravel()
            if vr.size >= 2:
                vmin, vmax = float(vr[0]), float(vr[1])

        # StructMetadata para a georreferência senoidal.
        meta_raw = h5["HDFEOS INFORMATION/StructMetadata.0"][()]
        meta = meta_raw.decode() if isinstance(meta_raw, bytes) else str(meta_raw)
        gm = _parse_struct_metadata(meta)

    # Máscara: fill e fora do valid_range -> NaN.
    raw[raw == fill] = np.nan
    raw[(raw < vmin) | (raw > vmax)] = np.nan
    evi = raw * scale

    ny, nx = evi.shape
    ulx, uly = gm["ul"]
    lrx, lry = gm["lr"]
    px = (lrx - ulx) / nx
    py = (lry - uly) / ny  # negativo

    xs = ulx + (np.arange(nx) + 0.5) * px
    ys = uly + (np.arange(ny) + 0.5) * py

    da = xr.DataArray(
        evi,
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="evi",
    )
    da = da.rio.write_crs(config.VIIRS_SINU_PROJ4)
    da = da.rio.set_nodata(np.nan)
    return da


# ---------------------------------------------------------------------------
# Cache por (tile, data): EVI reprojetado UMA vez para o grid do tile (EPSG:4326)
# ---------------------------------------------------------------------------
def viirs_tile_cog_path(h: int, v: int, date: dt.date) -> Path:
    return (
        Path(config.TILE_CACHE_DIR)
        / "viirs"
        / f"h{h:02d}v{v:02d}"
        / f"evi_h{h:02d}v{v:02d}_{date:%Y%m%d}.tif"
    )


def ensure_viirs_tile_cog(h: int, v: int, date: dt.date, h5_source) -> Path:
    """
    Garante o COG de EVI (EPSG:4326, grid do tile) para (tile, data),
    reprojetando o granule só na primeira vez. Reaproveitado por todos os
    hexágonos que tocam esse tile. ``h5_source`` é um caminho local ou um
    file-like (handle S3).
    """
    out = viirs_tile_cog_path(h, v, date)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    da = read_evi_tile(h5_source)
    grid = tiles.tile_grid(h, v)
    da_ll = da.rio.reproject_match(grid, resampling=Resampling.nearest)
    tmp = out.with_suffix(".tmp.tif")
    da_ll.rio.to_raster(tmp, driver="GTiff", tiled=True, compress="deflate", dtype="float32")
    tmp.replace(out)  # escrita atômica
    return out


def viirs_tile_dir(h: int, v: int) -> Path:
    return Path(config.TILE_CACHE_DIR) / "viirs" / f"h{h:02d}v{v:02d}"


def _tile_year_marker(h: int, v: int, year: int) -> Path:
    return viirs_tile_dir(h, v) / f".prepared_{year}"


def _date_from_cog(name: str) -> dt.date | None:
    m = _COG_DATE_RE.search(name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%d").date()


# ---------------------------------------------------------------------------
# PREPARE: baixa + reprojeta os COGs de um (tile, ano) UMA vez (toda a rede aqui)
# ---------------------------------------------------------------------------
def ensure_tile_year_viirs(h: int, v: int, year: int, cache_dir: Path | str = None) -> None:
    """
    Garante (uma vez) todos os COGs de EVI do tile (h, v) para o ano ``year``.

    Faz a busca no CMR, o download dos .h5 e a reprojeção -> COG, com novas
    tentativas em erros transitórios. Grava um marcador ``.prepared_{ano}`` para
    pular o trabalho (e toda a rede) nas próximas chamadas — é isso que torna o
    processamento por hexágono rápido e offline.
    """
    marker = _tile_year_marker(h, v, year)
    if marker.exists():
        return
    bbox = tiles.tile_latlon_bbox(h, v)
    if bbox is None:
        return

    grans = _with_retries(
        search_granules_bbox, bbox, f"{year}-01-01", f"{year}-12-31",
        what=f"search h{h:02d}v{v:02d} {year}",
    )
    grans = [g for g in grans if _safe_tile(g) == (h, v)]

    marker.parent.mkdir(parents=True, exist_ok=True)
    if not grans:
        marker.write_text("0 granules")
        return

    if use_s3_access():
        # Acesso direto ao S3 (in-region): copia/abre cada granule e gera o COG.
        cache = Path(cache_dir or config.VIIRS_CACHE_DIR)
        for g, src, cleanup in _iter_s3_sources(grans, cache):
            try:
                ensure_viirs_tile_cog(h, v, granule_date(g), src)
            except Exception as exc:
                tqdm.write(f"    [aviso] COG h{h:02d}v{v:02d} {granule_date(g)}: {exc!r}")
            finally:
                try:
                    cleanup()
                except Exception:
                    pass
    else:
        # Download HTTPS para o cache local (.h5) — comportamento histórico.
        path_by_ur = _with_retries(
            download_granules, grans, cache_dir,
            what=f"download h{h:02d}v{v:02d} {year}",
        )
        for g in grans:
            p = path_by_ur.get(granule_ur(g))
            if p is None or not Path(p).exists():
                continue
            try:
                ensure_viirs_tile_cog(h, v, granule_date(g), Path(p))
            except Exception as exc:
                tqdm.write(f"    [aviso] COG h{h:02d}v{v:02d} {granule_date(g)}: {exc!r}")
    marker.write_text(f"{len(grans)} granules")


# ---------------------------------------------------------------------------
# Construção do cubo diário (time, y, x) em EPSG:4326 — leitura SÓ do cache
# ---------------------------------------------------------------------------
def build_daily_cube(
    geometry,
    start: str,
    end: str,
    template: xr.DataArray,
    tiles_hv: list[tuple[int, int]] | None = None,
    cache_dir: Path | str = None,
) -> xr.DataArray:
    """
    Monta o cubo de EVI (time, y, x) no grid do ``template`` (EPSG:4326) lendo
    APENAS dos COGs cacheados por (tile, data).

    Garante o prepare do(s) tile(s) para o ano (cacheado/idempotente) e então,
    para cada data, recorta (janela) os COGs dos tiles que tocam o hexágono, faz
    o mosaico e alinha ao template — sem rede e sem reprojeção senoidal.

    Retorna None se não houver dados.
    """
    if tiles_hv is None:
        tiles_hv = tiles.tiles_for_geometry(geometry)
    year = int(str(start)[:4])
    for h, v in tiles_hv:
        ensure_tile_year_viirs(h, v, year, cache_dir)

    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()

    by_date: dict[dt.date, list[Path]] = defaultdict(list)
    for h, v in tiles_hv:
        for cog in viirs_tile_dir(h, v).glob("evi_*.tif"):
            d = _date_from_cog(cog.name)
            if d is None or not (start_d <= d <= end_d):
                continue
            by_date[d].append(cog)
    if not by_date:
        return None

    minx, miny, maxx, maxy = template.rio.bounds()
    pad = config.TARGET_RES_DEG * 2

    arrays = []
    dates = []
    for date in sorted(by_date):
        tile_arrays = []
        for cog in by_date[date]:
            try:
                # context manager + .load(): materializa e FECHA o dataset GDAL
                # (evita handles pendentes finalizados na saída do interpretador).
                with rioxarray.open_rasterio(cog, masked=True) as src:
                    da = src.squeeze("band", drop=True) if "band" in src.dims else src
                    da = da.rio.clip_box(minx - pad, miny - pad, maxx + pad, maxy + pad).load()
                tile_arrays.append(da)
            except Exception:  # tile não cobre a janela
                continue
        if not tile_arrays:
            continue
        mosaic = tile_arrays[0] if len(tile_arrays) == 1 else merge_arrays(tile_arrays)
        da_ll = mosaic.rio.reproject_match(template, resampling=Resampling.nearest)
        arrays.append(da_ll)
        dates.append(pd.Timestamp(date))

    if not arrays:
        return None

    cube = xr.concat(arrays, dim=pd.Index(dates, name="time"))
    cube = cube.sortby("time")
    cube.name = "evi"
    return cube
