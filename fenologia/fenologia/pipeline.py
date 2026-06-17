"""
Orquestração: agrupa hexágonos por "balde de tiles" VIIRS e processa cada
balde (todos os hexágonos, todos os anos) sem nenhum cache em disco.

Para cada balde de tiles (``tiles_for_geometry`` igual para um conjunto de
hexágonos vizinhos):
- monta um grid (``tiles.bucket_grid``) cobrindo a união dos tiles do balde;
- para cada ano, baixa/reprojeta o EVI VIIRS UMA VEZ para esse grid (em
  memória; o ``.h5`` de cada granule é temporário e apagado na hora — ver
  ``fenologia.viirs.build_tile_cube``) e lê o MapBiomas (coverage + safrinha)
  direto nesse grid (``fenologia.mapbiomas.read_mapbiomas_on_grid``, também
  sem cache em disco);
- recorta esses dados (em memória) à janela de cada hexágono do balde e
  extrai as séries de EVI por classe (``fenologia.extract``).
"""
from __future__ import annotations

import ctypes
import gc
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401
from shapely.geometry import mapping
from tqdm import tqdm

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _release_memory():
    """Force Python and glibc to return freed pages to the OS."""
    gc.collect()
    if _libc is not None:
        _libc.malloc_trim(0)

from . import config, extract, mapbiomas, tiles, viirs
from .grid import cell_to_polygon


def _clip_to_hex(da, geom):
    """Recorta um DataArray ao polígono do hexágono (fora -> nodata)."""
    return da.rio.clip([mapping(geom)], crs="EPSG:4326", drop=False, all_touched=True)


def _hex_window(hex_id: str) -> tuple[float, float, float, float]:
    geom = cell_to_polygon(hex_id)
    minx, miny, maxx, maxy = geom.bounds
    b = config.BBOX_BUFFER_DEG
    return minx - b, miny - b, maxx + b, maxy + b


def process_tile_bucket(
    hex_ids: Sequence[str],
    tiles_hv: Sequence[tuple[int, int]],
    years: Sequence[int] = tuple(config.YEARS),
    res_deg: float = config.TARGET_RES_DEG,
) -> dict[str, pd.DataFrame]:
    """
    Processa todos os ``hex_ids`` cujo conjunto de tiles VIIRS é ``tiles_hv``.

    Para cada ano, monta o cubo VIIRS e lê o MapBiomas (cobertura + safrinha)
    uma única vez no grid do balde, e então extrai as séries de EVI por classe
    para cada hexágono (formato wide).

    Retorna ``{hex_id: DataFrame wide}`` (DataFrame vazio com o schema wide se
    o hexágono não tiver dados).
    """
    template = tiles.bucket_grid(tiles_hv, res_deg)
    label = tiles.bucket_label(tiles_hv)

    frames_by_hex: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for year in years:
        with tempfile.TemporaryDirectory(prefix="fenologia_") as tmp:
            cube = viirs.build_tile_cube(list(tiles_hv), year, template, Path(tmp), f"{label} {year}")
        if cube is None:
            continue

        # Amarra cada data ao seu ano-calendário: a janela de busca pode trazer
        # composições da virada do ano anterior; ficamos só com as do ano `year`
        # (assim cada data é processada uma única vez, com o MapBiomas do ano
        # correspondente).
        cube = cube.sel(time=cube["time"].dt.year == year)
        if cube.sizes["time"] == 0:
            continue

        cov = mapbiomas.read_mapbiomas_on_grid(year, "coverage", template)
        sec = mapbiomas.read_mapbiomas_on_grid(year, "second_crop", template)

        for hex_id in hex_ids:
            geom = cell_to_polygon(hex_id)
            window = _hex_window(hex_id)

            cube_hex = cube.rio.clip_box(*window)
            cov_hex = _clip_to_hex(cov.rio.clip_box(*window), geom)
            sec_hex = _clip_to_hex(sec.rio.clip_box(*window), geom)

            df_cov = extract.extract_class_series(
                cube_hex, cov_hex, config.COVERAGE_AGRI_CLASSES,
                hex_id=hex_id, year=year, source="coverage", res_deg=res_deg,
            )
            df_sec = extract.extract_class_series(
                cube_hex, sec_hex, config.SECOND_CROP_CLASSES,
                hex_id=hex_id, year=year, source="second_crop", res_deg=res_deg,
            )
            if len(df_cov):
                frames_by_hex[hex_id].append(df_cov)
            if len(df_sec):
                frames_by_hex[hex_id].append(df_sec)

        # Libera explicitamente o cubo e os MapBiomas antes do próximo ano,
        # e força glibc a devolver páginas ao SO (evita crescimento de RSS).
        del cube, cov, sec
        _release_memory()

    result: dict[str, pd.DataFrame] = {}
    for hex_id in hex_ids:
        frames = frames_by_hex.get(hex_id)
        if frames:
            result[hex_id] = extract.to_wide(pd.concat(frames, ignore_index=True))
        else:
            result[hex_id] = extract.to_wide(None)
    return result


# ---------------------------------------------------------------------------
# Saída: um parquet por balde (em ``_parts/``) + concatenação final
# ---------------------------------------------------------------------------
def _part_path(output_dir: Path, label: str) -> Path:
    return output_dir / "_parts" / f"balde_{label}.parquet"


def concat_parts(output_dir: Path | str, filename: str = "evi_brazil.parquet") -> Path:
    """
    Concatena todos os parquets de ``_parts/`` (um por balde de tiles, já
    escrito) num único parquet final ``output_dir/filename``.

    Pode ser chamada a qualquer momento — inclusive com ``run`` ainda em
    andamento — para obter um snapshot parcial do resultado.
    """
    output_dir = Path(output_dir)
    parts = sorted((output_dir / "_parts").glob("*.parquet"))
    frames = [df for df in (pd.read_parquet(p) for p in parts) if len(df)]
    if frames:
        full = pd.concat(frames, ignore_index=True)
        full = full.sort_values(["id_hexagono", "data"]).reset_index(drop=True)
    else:
        full = pd.DataFrame(columns=extract.full_wide_columns())
    out_path = output_dir / filename
    full.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Execução em lote
# ---------------------------------------------------------------------------
def run(
    boundary: Optional[gpd.GeoDataFrame] = None,
    bbox: Optional[Sequence[float]] = None,
    years: Sequence[int] = tuple(config.YEARS),
    output_dir: Path | str = None,
    hex_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    resume: bool = True,
) -> None:
    """
    Roda o pipeline para um conjunto de hexágonos.

    - Se ``hex_ids`` for dado, usa essa lista; senão gera o grid a partir de
      ``boundary`` (GeoDataFrame) ou ``bbox``.
    - ``limit`` processa apenas os N primeiros (útil p/ teste).
    - ``resume`` pula baldes cujo parquet em ``_parts/`` já existe.

    Os hexágonos são agrupados em "baldes" pelo conjunto de tiles VIIRS que os
    cobrem (``tiles.tiles_for_geometry``); cada balde é processado de uma vez
    (sem cache em disco — ver ``process_tile_bucket``) e o resultado de TODOS
    os seus hexágonos (mesmo os sem agricultura, com 0 linhas) é gravado em
    ``output_dir/_parts/balde_<label>.parquet`` — esse arquivo é o marcador de
    "balde concluído" usado pelo ``resume``. Ao final, todos os baldes são
    concatenados num único ``output_dir/evi_brazil.parquet``
    (``concat_parts``).
    """
    from .grid import build_h3_grid

    output_dir = Path(output_dir or config.OUTPUT_DIR)
    (output_dir / "_parts").mkdir(parents=True, exist_ok=True)

    if hex_ids is None:
        grid = build_h3_grid(boundary=boundary, bbox=bbox, resolution=config.H3_RESOLUTION)
        hex_ids = grid["h3"].tolist()
    hex_ids = list(hex_ids)
    if limit:
        hex_ids = hex_ids[:limit]

    buckets: dict[tuple[tuple[int, int], ...], list[str]] = defaultdict(list)
    for hex_id in hex_ids:
        tiles_hv = tuple(sorted(tiles.tiles_for_geometry(cell_to_polygon(hex_id))))
        buckets[tiles_hv].append(hex_id)

    n_written = n_skip = n_empty = n_err = 0
    bar = tqdm(buckets.items(), desc="Tiles", unit="balde")
    for tiles_hv, bucket_hex_ids in bar:
        label = tiles.bucket_label(tiles_hv)
        part_path = _part_path(output_dir, label)

        if resume and part_path.exists():
            n_skip += len(bucket_hex_ids)
            bar.set_postfix_str(f"{label}: já existia (cache)")
            continue

        t0 = time.time()
        try:
            results = process_tile_bucket(bucket_hex_ids, tiles_hv, years=years)
        except Exception as exc:
            n_err += len(bucket_hex_ids)
            tqdm.write(f"[{label}] ERRO {exc!r}")
            continue
        dt_s = time.time() - t0

        frames = [df for df in results.values() if len(df)]
        n_written += len(frames)
        n_empty += len(bucket_hex_ids) - len(frames)
        bucket_df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=extract.full_wide_columns())
        )
        bucket_df.to_parquet(part_path, index=False)
        bar.set_postfix_str(f"{label}: {len(bucket_hex_ids)} hex, {dt_s:.1f}s")

    out_path = concat_parts(output_dir)
    print(
        f"Concluído ({len(hex_ids)} hexágonos, {len(buckets)} baldes de tiles): "
        f"{n_written} com agricultura, {n_skip} hex em baldes já concluídos, "
        f"{n_empty} sem agricultura, {n_err} erros. Saída em {out_path}"
    )
    if n_skip and not n_written and not n_empty and not n_err:
        print("Todos os baldes já estavam concluídos. "
              "Use --no-resume para refazer, ou apague os parquets em "
              f"{output_dir / '_parts'}.")
