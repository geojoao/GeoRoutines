"""
Acesso aos dados de EVI do VIIRS (produto VNP13A1, 500 m, composição de 16 dias)
da NASA LP DAAC, via biblioteca ``earthaccess``.

Particularidades tratadas aqui:
- Os granules vêm em HDF-EOS5 (.h5) na grade senoidal por tile (h/v). Em vez de
  depender do driver HDF5 do GDAL (que normalmente NÃO vem nos wheels do
  rasterio), lemos o array com ``h5py`` e montamos a georreferência senoidal a
  partir do ``StructMetadata.0``. Depois reprojetamos para EPSG:4326.
- Os granules são abertos via ``earthaccess.open`` (S3 direto quando o
  ambiente está in-region, na mesma AWS region da LP DAAC; HTTPS caso
  contrário) e copiados para um arquivo temporário só durante a leitura — o
  arquivo é apagado imediatamente depois. Cada granule aberto gera um log
  indicando se o acesso foi via S3 ou HTTPS.
- Para cobrir um "balde" de tiles, fazemos o mosaico (``combine_first``) dos
  tiles da mesma data de composição já reprojetados para o grid do balde.
"""
from __future__ import annotations

import datetime as dt
import re
import shutil
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import earthaccess
import h5py
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registra o accessor .rio)
import xarray as xr
from rasterio.enums import Resampling
from tqdm import tqdm

from . import config, tiles

# Regex para extrair a data de composição (AYYYYDDD) do GranuleUR.
_DATE_RE = re.compile(r"\.A(\d{4})(\d{3})\.")


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

    # fsspec.config.conf pode ter uma entrada 'gcs' (gerada pelo gcsfs/MapBiomas).
    # O earthaccess.open usa fsspec.config.conf como open_kwargs padrão quando
    # nenhum argumento é passado, o que faz a chave 'gcs' chegar ao filesystem
    # HTTP e quebrar com TypeError.  Remove antes de autenticar.
    try:
        import fsspec.config as _fsc
        _fsc.conf.pop("gcs", None)
    except Exception:
        pass

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
# Busca
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


def _safe_tile(granule) -> tuple[int, int] | None:
    try:
        return tiles.parse_tile(granule_ur(granule))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Acesso direto: S3 (in-region) ou HTTPS, via earthaccess.open
# ---------------------------------------------------------------------------
def open_granule(granule):
    """
    Abre o granule via ``earthaccess.open`` (S3 direto se o ambiente estiver
    in-region na AWS region da LP DAAC; HTTPS caso contrário), com retries.
    """
    authenticate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        files = _with_retries(
            earthaccess.open, [granule], what=f"open {granule_ur(granule)}"
        )
    return files[0]


def _access_kind(fobj) -> tuple[str, type]:
    """
    Identifica se ``fobj`` (um ``EarthAccessFile``) está apoiado num
    ``s3fs.S3File`` (acesso direto ao S3) ou num file-like HTTPS.

    ``fobj.__class__`` é *proxied* pelo ``EarthAccessFile`` para a classe do
    file-like real (``type(fobj)`` continuaria sendo ``EarthAccessFile``).
    """
    cls = fobj.__class__
    mod = getattr(cls, "__module__", "")
    kind = "s3" if mod.startswith("s3fs") else "https"
    return kind, cls


def _fetch_granule_to_temp(granule, tmp_dir: Path, label: str) -> Path:
    """
    Abre o granule (S3 ou HTTPS) e copia para ``tmp_dir`` (apagado pelo
    chamador logo após a leitura). Loga o tipo de acesso usado.
    """
    fobj = open_granule(granule)
    kind, cls = _access_kind(fobj)
    tqdm.write(
        f"    [{label}] {granule_ur(granule)}: acesso via "
        f"{kind.upper()} ({cls.__module__}.{cls.__name__})"
    )
    dest = tmp_dir / f"{granule_ur(granule)}.h5"
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(fobj, out)
    finally:
        try:
            fobj.close()
        except Exception:
            pass
    return dest


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

    ``h5_source`` pode ser um caminho (str/Path) OU um objeto file-like —
    ``h5py.File`` aceita ambos.
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
# Cubo (time, y, x) para um "balde" de tiles — tudo em memória
# ---------------------------------------------------------------------------
def build_tile_cube(
    tiles_hv: list[tuple[int, int]],
    year: int,
    template: xr.DataArray,
    tmp_dir: Path,
    label: str,
) -> xr.DataArray | None:
    """
    Monta o cubo de EVI (time, y, x) no grid ``template`` (EPSG:4326, cobrindo
    o balde de tiles ``tiles_hv``) para o ano ``year``.

    Busca os granules VNP13A1 no CMR, abre cada um (S3/HTTPS, logado), lê o
    EVI senoidal, reprojeta direto para ``template`` e apaga o ``.h5``
    temporário. Tiles diferentes do mesmo dia são mosaicados
    (``combine_first``). Nada é persistido em disco além do ``.h5`` de cada
    granule, apagado imediatamente após a leitura.

    Retorna ``None`` se não houver granules/dados.
    """
    bbox = template.rio.bounds()
    grans = _with_retries(
        search_granules_bbox, bbox, f"{year}-01-01", f"{year}-12-31",
        what=f"search {label} {year}",
    )
    tiles_set = set(tiles_hv)
    grans = [g for g in grans if _safe_tile(g) in tiles_set]
    if not grans:
        return None

    by_date: dict[dt.date, list] = defaultdict(list)
    for g in grans:
        by_date[granule_date(g)].append(g)

    # Pre-download all granules in parallel (IO-bound → threads safe here)
    all_grans = [g for gs in by_date.values() for g in gs]

    def _dl(g):
        try:
            path = _fetch_granule_to_temp(g, tmp_dir, f"{label}")
            return granule_ur(g), path
        except Exception as exc:
            tqdm.write(f"    [erro download] {granule_ur(g)}: {exc!r}")
            return granule_ur(g), None

    n_workers = min(8, max(1, len(all_grans)))
    prefetched: dict[str, Path | None] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        for ur, path in exe.map(_dl, all_grans):
            prefetched[ur] = path

    arrays = []
    dates = []
    for date in sorted(by_date):
        tile_arrays = []
        for g in by_date[date]:
            h5_path = prefetched.get(granule_ur(g))
            if h5_path is None:
                continue
            try:
                da = read_evi_tile(h5_path)
                da_ll = da.rio.reproject_match(template, resampling=Resampling.nearest)
                tile_arrays.append(da_ll)
            except Exception as exc:
                tqdm.write(f"    [aviso] {label} {date} {granule_ur(g)}: {exc!r}")
            finally:
                if h5_path is not None:
                    h5_path.unlink(missing_ok=True)
        if not tile_arrays:
            continue
        mosaic = tile_arrays[0]
        for extra in tile_arrays[1:]:
            mosaic = mosaic.combine_first(extra)
        arrays.append(mosaic)
        dates.append(pd.Timestamp(date))

    if not arrays:
        return None

    cube = xr.concat(arrays, dim=pd.Index(dates, name="time"))
    cube = cube.sortby("time")
    cube.name = "evi"
    return cube
