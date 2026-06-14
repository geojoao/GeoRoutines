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
    label = "+".join(f"h{h:02d}v{v:02d}" for h, v in tiles_hv)

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

        # `cube`, `cov`, `sec` saem de escopo aqui (próximo ano descarta e
        # reconstrói) — nada fica retido em memória entre baldes/anos.

    result: dict[str, pd.DataFrame] = {}
    for hex_id in hex_ids:
        frames = frames_by_hex.get(hex_id)
        if frames:
            result[hex_id] = extract.to_wide(pd.concat(frames, ignore_index=True))
        else:
            result[hex_id] = extract.to_wide(None)
    return result


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
    - ``resume`` pula hexágonos cujo parquet já existe.

    Os hexágonos são agrupados em "baldes" pelo conjunto de tiles VIIRS que os
    cobrem (``tiles.tiles_for_geometry``); cada balde é processado de uma vez
    (sem cache em disco — ver ``process_tile_bucket``) e gera um parquet por
    hexágono em ``output_dir``.
    """
    from .grid import build_h3_grid

    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        label = "+".join(f"h{h:02d}v{v:02d}" for h, v in tiles_hv)

        pending = [
            hex_id for hex_id in bucket_hex_ids
            if not (resume and (output_dir / f"evi_{hex_id}.parquet").exists())
        ]
        if not pending:
            n_skip += len(bucket_hex_ids)
            bar.set_postfix_str(f"{label}: {len(bucket_hex_ids)} já existiam (cache)")
            continue

        t0 = time.time()
        try:
            results = process_tile_bucket(pending, tiles_hv, years=years)
        except Exception as exc:
            n_err += len(pending)
            tqdm.write(f"[{label}] ERRO {exc!r}")
            continue
        dt_s = time.time() - t0

        n_skip += len(bucket_hex_ids) - len(pending)
        for hex_id, df in results.items():
            out_path = output_dir / f"evi_{hex_id}.parquet"
            if len(df):
                df.to_parquet(out_path, index=False)
                n_written += 1
            else:
                n_empty += 1
        bar.set_postfix_str(f"{label}: {len(pending)} hex, {dt_s:.1f}s")

    print(
        f"Concluído ({len(hex_ids)} hexágonos, {len(buckets)} baldes de tiles): "
        f"{n_written} novos, {n_skip} já existiam, {n_empty} sem agricultura, "
        f"{n_err} erros. Saída em {output_dir}"
    )
    if n_skip and not n_written and not n_empty:
        print("Todos os hexágonos já estavam processados. "
              "Use --no-resume para refazer, ou apague os parquets em questão.")
