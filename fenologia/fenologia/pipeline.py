"""
Orquestração: processa um hexágono (todos os anos) e roda o grid completo.

Etapas (para dividir o trabalho e maximizar reuso de cache):
- ``prepare_tiles`` / ``prepare_mapbiomas``: pré-geram os COGs por tile (passo
  pesado e reaproveitável), com progress bar.
- ``process_hexagon`` / ``run``: leitura por janela (barata) + extração.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401
from shapely.geometry import mapping
from tqdm import tqdm

from . import config, extract, mapbiomas, tiles, viirs
from .grid import cell_to_polygon
from .rasterutils import make_target_grid


def _clip_to_hex(da, geom):
    """Recorta um DataArray ao polígono do hexágono (fora -> nodata)."""
    return da.rio.clip([mapping(geom)], crs="EPSG:4326", drop=False, all_touched=True)


def process_hexagon(
    hex_id: str,
    years: Sequence[int] = tuple(config.YEARS),
    cache_dir: Path | str = None,
    res_deg: float = config.TARGET_RES_DEG,
) -> pd.DataFrame:
    """
    Processa um único hexágono H3: para cada ano, monta o cubo VIIRS, lê o
    MapBiomas (cobertura + safrinha) do cache por tile e extrai as séries de EVI
    por classe (formato wide).
    """
    geom = cell_to_polygon(hex_id)
    minx, miny, maxx, maxy = geom.bounds
    b = config.BBOX_BUFFER_DEG
    bounds = (minx - b, miny - b, maxx + b, maxy + b)
    template = make_target_grid(bounds, res_deg)

    # Tiles que cobrem o hexágono (mesmos para VIIRS e MapBiomas).
    tiles_hv = tiles.tiles_for_geometry(geom)

    frames = []
    for year in years:
        cube = viirs.build_daily_cube(
            geom, f"{year}-01-01", f"{year}-12-31", template,
            tiles_hv=tiles_hv, cache_dir=cache_dir,
        )
        if cube is None:
            continue

        # Amarra cada data ao seu ano-calendário: a janela de busca pode trazer
        # composições da virada do ano anterior; ficamos só com as do ano `year`
        # (assim cada data é processada uma única vez, com o MapBiomas do ano
        # correspondente).
        cube = cube.sel(time=cube["time"].dt.year == year)
        if cube.sizes["time"] == 0:
            continue

        # MapBiomas no grid do VIIRS (do cache por tile), recortado ao hexágono.
        cov = _clip_to_hex(
            mapbiomas.read_mapbiomas_on_grid(year, "coverage", template, tiles_hv), geom
        )
        sec = _clip_to_hex(
            mapbiomas.read_mapbiomas_on_grid(year, "second_crop", template, tiles_hv), geom
        )

        df_cov = extract.extract_class_series(
            cube, cov, config.COVERAGE_AGRI_CLASSES,
            hex_id=hex_id, year=year, source="coverage", res_deg=res_deg,
        )
        df_sec = extract.extract_class_series(
            cube, sec, config.SECOND_CROP_CLASSES,
            hex_id=hex_id, year=year, source="second_crop", res_deg=res_deg,
        )
        if len(df_cov):
            frames.append(df_cov)
        if len(df_sec):
            frames.append(df_sec)

    if not frames:
        return extract.to_wide(None)  # DataFrame vazio com o schema wide
    df_long = pd.concat(frames, ignore_index=True)
    return extract.to_wide(df_long)


# ---------------------------------------------------------------------------
# Etapa de preparação (pesada, uma vez) — caches por tile
# ---------------------------------------------------------------------------
def prepare_mapbiomas(tiles_hv: Iterable[tuple[int, int]], years: Sequence[int]) -> None:
    """Pré-gera os COGs do MapBiomas (cobertura + safrinha) por (tile, ano)."""
    tiles_hv = list(tiles_hv)
    jobs = [(h, v, y, k) for (h, v) in tiles_hv for y in years for k in ("coverage", "second_crop")]
    for h, v, y, k in tqdm(jobs, desc="MapBiomas (tile,ano,tipo)", unit="job"):
        try:
            mapbiomas.ensure_mapbiomas_tile_cog(h, v, y, k)
        except Exception as exc:
            tqdm.write(f"[mapbiomas h{h:02d}v{v:02d} {y} {k}] ERRO {exc!r}")


def prepare_viirs(
    tiles_hv: Iterable[tuple[int, int]],
    years: Sequence[int],
    cache_dir: Path | str = None,
) -> None:
    """
    Pré-gera os COGs de EVI do VIIRS por (tile, ano): aqui acontece TODA a rede
    (busca CMR + download dos .h5) e a reprojeção -> COG, uma única vez por tile.
    Depois, o processamento por hexágono é local e rápido.
    """
    tiles_hv = list(tiles_hv)
    jobs = [(h, v, y) for (h, v) in tiles_hv for y in years]
    for h, v, y in tqdm(jobs, desc="VIIRS (tile,ano)", unit="job"):
        try:
            viirs.ensure_tile_year_viirs(h, v, y, cache_dir)
        except Exception as exc:
            tqdm.write(f"[viirs h{h:02d}v{v:02d} {y}] ERRO {exc!r}")


def tiles_for_hexagons(hex_ids: Iterable[str]) -> list[tuple[int, int]]:
    """União dos tiles que cobrem uma lista de hexágonos."""
    seen: set[tuple[int, int]] = set()
    for hid in hex_ids:
        seen.update(tiles.tiles_for_geometry(cell_to_polygon(hid)))
    return sorted(seen)


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
    prepare: bool = True,
) -> None:
    """
    Roda o pipeline para um conjunto de hexágonos.

    - Se ``hex_ids`` for dado, usa essa lista; senão gera o grid a partir de
      ``boundary`` (GeoDataFrame) ou ``bbox``.
    - ``limit`` processa apenas os N primeiros (útil p/ teste).
    - ``resume`` pula hexágonos cujo parquet já existe.
    - ``prepare`` pré-gera os COGs do MapBiomas por tile antes do loop.

    Salva um parquet por hexágono em ``output_dir`` e mostra progress bar com o
    tempo médio por hexágono.
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

    # Pré-geração (passo pesado, uma vez por tile e reaproveitado por todos os
    # hexágonos): VIIRS (toda a rede aqui) + MapBiomas (mode-resample 30m->463m).
    if prepare:
        tiles_hv = tiles_for_hexagons(hex_ids)
        n_years = len(list(years))
        print(f"Preparando {len(tiles_hv)} tiles x {n_years} anos (VIIRS + MapBiomas)...")
        prepare_viirs(tiles_hv, years)
        prepare_mapbiomas(tiles_hv, years)

    n_written = n_skip = n_empty = n_err = 0
    bar = tqdm(hex_ids, desc="Hexágonos", unit="hex")
    for hex_id in bar:
        out_path = output_dir / f"evi_{hex_id}.parquet"
        if resume and out_path.exists():
            n_skip += 1
            bar.set_postfix_str(f"{hex_id}: já existe (cache)")
            continue
        t0 = time.time()
        try:
            df = process_hexagon(hex_id, years=years)
        except Exception as exc:
            n_err += 1
            tqdm.write(f"[{hex_id}] ERRO {exc!r}")
            continue
        dt_s = time.time() - t0
        if len(df):
            df.to_parquet(out_path, index=False)
            n_written += 1
            bar.set_postfix_str(f"{hex_id}: {len(df)} linhas, {dt_s:.1f}s/hex")
        else:
            n_empty += 1
            bar.set_postfix_str(f"{hex_id}: sem agricultura, {dt_s:.1f}s/hex")

    print(
        f"Concluído ({len(hex_ids)} hexágonos): {n_written} novos, "
        f"{n_skip} já existiam, {n_empty} sem agricultura, {n_err} erros. "
        f"Saída em {output_dir}"
    )
    if n_skip and not n_written and not n_empty:
        print("Todos os hexágonos já estavam processados. "
              "Use --no-resume para refazer, ou apague os parquets em questão.")
