"""
Acesso aos dados de EVI do VIIRS (produto VNP13A1, 500 m, composição de 16 dias)
da NASA LP DAAC, via biblioteca ``earthaccess``.

Particularidades tratadas aqui:
- Os granules vêm em HDF-EOS5 (.h5) na grade senoidal por tile (h/v). Em vez de
  depender do driver HDF5 do GDAL (que normalmente NÃO vem nos wheels do
  rasterio), lemos o array com ``h5py`` e montamos a georreferência senoidal a
  partir do ``StructMetadata.0``. Depois reprojetamos para EPSG:4326.
- Os granules são abertos priorizando o **S3 direto** (in-region us-west-2):
  construímos um ``s3fs`` com credenciais temporárias do Earthdata e abrimos o
  link ``s3://`` do granule, sem depender da auto-detecção de região do
  earthaccess (que consulta o IMDS da EC2 e falha em pods Kubernetes, caindo
  para HTTPS mesmo in-region). O modo é configurável por ``VIIRS_ACCESS_MODE``
  (``auto``/``s3``/``https``). Cada granule é copiado para um arquivo temporário
  só durante a leitura — apagado imediatamente depois — e gera um log indicando
  se o acesso foi via S3 ou HTTPS.
- Para cobrir um "balde" de tiles, fazemos o mosaico (``combine_first``) dos
  tiles da mesma data de composição já reprojetados para o grid do balde. O
  cubo (time, y, x) é escrito direto num buffer pré-alocado, fatia a fatia,
  para não duplicar o ano inteiro em RAM (evita OOM).
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
# Acesso direto: S3 (in-region) ou HTTPS
# ---------------------------------------------------------------------------
# Em vez de deixar o ``earthaccess.open`` decidir por auto-detecção de região
# (que consulta o IMDS da EC2 e falha em pods Kubernetes -> cai para HTTPS
# mesmo in-region), abrimos os granules explicitamente pelo S3 direto usando um
# ``s3fs`` construído com credenciais temporárias do Earthdata. Só caímos para
# HTTPS quando o S3 realmente não está disponível (fora da região) e o modo
# permite (``auto``).
_S3FS = None
_S3FS_TS = 0.0
_S3FS_TTL = 45 * 60.0  # renova as credenciais bem antes de expirarem (~1h)


def _reset_s3_filesystem() -> None:
    global _S3FS, _S3FS_TS
    _S3FS = None
    _S3FS_TS = 0.0


def _new_s3_filesystem(results=None):
    """Cria um ``s3fs.S3FileSystem`` com credenciais S3 temporárias da LP DAAC."""
    authenticate()
    prov = config.VIIRS_S3_PROVIDER
    attempts = [{"provider": prov}, {"daac": prov}]
    if results:
        attempts.append({"results": results})
    last = None
    for kwargs in attempts:
        try:
            return earthaccess.get_s3_filesystem(**kwargs)
        except Exception as exc:  # assinatura varia entre versões do earthaccess
            last = exc
    raise last or RuntimeError("earthaccess.get_s3_filesystem indisponível")


def _s3_filesystem(results=None):
    """Devolve um ``s3fs`` cacheado, renovando as credenciais periodicamente."""
    global _S3FS, _S3FS_TS
    now = time.time()
    if _S3FS is None or (now - _S3FS_TS) > _S3FS_TTL:
        _S3FS = _new_s3_filesystem(results=results)
        _S3FS_TS = now
    return _S3FS


def _direct_s3_url(granule) -> str | None:
    """Extrai o link S3 direto (``s3://.../*.h5``) do granule, se houver."""
    try:
        links = granule.data_links(access="direct")
    except Exception:
        links = None
    if not links:
        return None
    for link in links:
        if link.startswith("s3://") and link.lower().endswith(".h5"):
            return link
    for link in links:
        if link.startswith("s3://"):
            return link
    return None


def _open_s3(granule):
    """Tenta abrir o granule direto do S3; devolve ``None`` se indisponível."""
    url = _direct_s3_url(granule)
    if not url:
        return None
    try:
        fs = _s3_filesystem(results=[granule])
        return _with_retries(
            fs.open, url, mode="rb",
            what=f"s3 open {granule_ur(granule)}", attempts=3,
        )
    except Exception as exc:
        tqdm.write(
            f"    [s3 indisponível -> https] {granule_ur(granule)}: {exc!r}"
        )
        _reset_s3_filesystem()  # força renovar credenciais na próxima tentativa
        return None


def _open_https(granule):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        files = _with_retries(
            earthaccess.open, [granule], what=f"open {granule_ur(granule)}"
        )
    return files[0]


def open_granule(granule):
    """
    Abre o granule priorizando o **S3 direto** (in-region us-west-2) conforme
    ``config.VIIRS_ACCESS_MODE``:

    - ``"s3"``   -> só S3 (erro se indisponível);
    - ``"https"``-> só HTTPS;
    - ``"auto"`` -> S3 direto e, se falhar, HTTPS (padrão).
    """
    authenticate()
    mode = config.VIIRS_ACCESS_MODE
    if mode in ("auto", "s3"):
        fobj = _open_s3(granule)
        if fobj is not None:
            return fobj
        if mode == "s3":
            raise RuntimeError(
                f"acesso S3 exigido (FENOLOGIA_ACCESS=s3) mas indisponível para "
                f"{granule_ur(granule)}"
            )
    return _open_https(granule)


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

    Busca os granules VNP13A1 no CMR, abre cada um (S3 direto/HTTPS, logado),
    lê o EVI senoidal, reprojeta direto para ``template`` e apaga o ``.h5``
    temporário. Tiles diferentes do mesmo dia são mosaicados
    (``combine_first``) e escritos numa fatia de um buffer float32
    pré-alocado (sem ``xr.concat`` de uma lista, para não duplicar o cubo em
    RAM). Os downloads usam uma janela de prefetch limitada
    (``VIIRS_PREFETCH_DATES``), então não seguramos o ano inteiro de granules
    em disco de uma vez. Nada é persistido em disco além do ``.h5`` de cada
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
    dates_sorted = sorted(by_date)

    # --- Cubo pré-alocado -------------------------------------------------
    # Escrevemos cada data direto numa fatia de um buffer float32 já alocado no
    # tamanho do grid do balde, em vez de acumular uma lista de DataArrays e
    # fazer ``xr.concat`` no fim (que duplicava o cubo inteiro em RAM no pico).
    # Assim o teto de memória é ~1x o cubo + a área de trabalho de UMA data.
    ny = template.sizes["y"]
    nx = template.sizes["x"]
    data = np.full((len(dates_sorted), ny, nx), np.nan, dtype="float32")
    kept_dates: list[pd.Timestamp] = []
    kept = 0

    # --- Download com janela de prefetch limitada -------------------------
    # Baixar os granules é IO-bound (threads são seguras). Mantemos só algumas
    # datas "em voo" (``VIIRS_PREFETCH_DATES``) para sobrepor rede e CPU sem
    # segurar em disco/memória o ano inteiro de granules de uma vez.
    def _dl_one(g):
        try:
            return _fetch_granule_to_temp(g, tmp_dir, f"{label}")
        except Exception as exc:
            tqdm.write(f"    [erro download] {granule_ur(g)}: {exc!r}")
            return None

    with ThreadPoolExecutor(max_workers=config.VIIRS_DL_WORKERS) as exe:
        futures_by_date: dict[dt.date, dict[str, "object"]] = {}

        def _submit(date):
            futures_by_date[date] = {
                granule_ur(g): exe.submit(_dl_one, g) for g in by_date[date]
            }

        next_i = 0
        for _ in range(min(config.VIIRS_PREFETCH_DATES, len(dates_sorted))):
            _submit(dates_sorted[next_i])
            next_i += 1

        for date in dates_sorted:
            futs = futures_by_date.pop(date)
            # mantém a janela de prefetch cheia
            if next_i < len(dates_sorted):
                _submit(dates_sorted[next_i])
                next_i += 1

            mosaic = None
            for g in by_date[date]:
                h5_path = futs[granule_ur(g)].result()
                if h5_path is None:
                    continue
                try:
                    da = read_evi_tile(h5_path)
                    da_ll = da.rio.reproject_match(
                        template, resampling=Resampling.nearest
                    )
                    del da
                    mosaic = da_ll if mosaic is None else mosaic.combine_first(da_ll)
                    del da_ll
                except Exception as exc:
                    tqdm.write(f"    [aviso] {label} {date} {granule_ur(g)}: {exc!r}")
                finally:
                    h5_path.unlink(missing_ok=True)

            if mosaic is None:
                continue
            arr = np.asarray(mosaic.transpose("y", "x").values, dtype="float32")
            del mosaic
            if arr.shape != (ny, nx):
                tqdm.write(
                    f"    [aviso] {label} {date}: shape inesperado {arr.shape}, "
                    "pulando"
                )
                continue
            data[kept] = arr
            kept_dates.append(pd.Timestamp(date))
            kept += 1
            del arr

    if kept == 0:
        return None

    cube = xr.DataArray(
        data[:kept],
        coords={
            "time": pd.Index(kept_dates, name="time"),
            "y": template["y"],
            "x": template["x"],
        },
        dims=("time", "y", "x"),
        name="evi",
    )
    cube = cube.rio.write_crs(template.rio.crs)
    cube = cube.rio.set_nodata(np.nan)
    cube = cube.sortby("time")
    return cube
