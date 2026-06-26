#!/usr/bin/env python3
"""
Extrai séries temporais EVI no nível do pixel para um único hexágono H3.

Somente pixels classificados como soja (código 39) pelo MapBiomas em pelo
menos um dos anos processados.  Para esses pixels, o EVI é extraído em
TODOS os anos configurados (contexto temporal completo).

Uso:
    uv run python extract_pixel_series.py 85815527fffffff
    uv run python extract_pixel_series.py 85815527fffffff --years 2021 2022 2023

Sugestão de hexágono para fronteira agrícola MATOPIBA (Tocantins):
    85815527fffffff  (lat≈-8.09, lon≈-47.68, 7 ciclos safra+safrinha)

Saída: data/output/pixel_series/{hex_id}.parquet
  lon        float   longitude do centro do pixel (°)
  lat        float   latitude do centro do pixel (°)
  data       date    data da composição VIIRS
  evi        float   EVI físico (0–1, NaN se inválido)
  soja_ano   bool    pixel estava em soja NAQUELE ano (MapBiomas)
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import mapping

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from fenologia import config, mapbiomas, tiles, viirs
from fenologia.grid import cell_to_polygon

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

SOJA_CODE    = 39
OUTPUT_DIR   = config.OUTPUT_DIR / "pixel_series"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_box_hex(da: xr.DataArray, geom, buffer: float = config.BBOX_BUFFER_DEG) -> xr.DataArray:
    """Clip data array to the hexagon bounding box + buffer."""
    import rioxarray  # noqa: F401
    minx, miny, maxx, maxy = geom.bounds
    return da.rio.clip_box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)


def _clip_poly(da: xr.DataArray, geom) -> xr.DataArray:
    """Mask pixels outside the hexagon polygon (NaN outside, retain shape)."""
    import rioxarray  # noqa: F401
    return da.rio.clip([mapping(geom)], crs="EPSG:4326", drop=False, all_touched=True)


def _read_soy_mask(year: int, template: xr.DataArray, geom) -> np.ndarray:
    """Returns a 2-D boolean array (y, x) — True where MapBiomas == soja."""
    cov = mapbiomas.read_mapbiomas_on_grid(year, "coverage", template)
    cov_clipped = _clip_poly(_clip_box_hex(cov, geom), geom)
    return (np.asarray(cov_clipped) == SOJA_CODE)  # (y, x)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_pixel_series(
    hex_id: str,
    years: list[int] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """
    Extrai séries EVI por pixel para o hexágono ``hex_id``.

    Retorna o Path do parquet gerado (ou None se nenhum dado encontrado).
    """
    if years is None:
        years = list(config.YEARS)

    import rioxarray  # noqa: F401

    geom = cell_to_polygon(hex_id)
    tiles_hv = sorted(tiles.tiles_for_geometry(geom))
    label    = tiles.bucket_label(tiles_hv)
    template = tiles.bucket_grid(tiles_hv, config.TARGET_RES_DEG)

    log.info(f"Hexágono {hex_id}  tiles={label}  anos={years}")

    # ── Passo 1: Lê máscaras MapBiomas para todos os anos ──────────────────
    # (leve: apenas range-requests no GeoTIFF remoto)
    log.info("Lendo máscaras MapBiomas (soja)...")
    soy_masks: dict[int, np.ndarray] = {}
    ref_coords: tuple[np.ndarray, np.ndarray] | None = None  # (y_arr, x_arr)

    for year in years:
        cov = mapbiomas.read_mapbiomas_on_grid(year, "coverage", template)
        cov_clipped = _clip_poly(_clip_box_hex(cov, geom), geom)
        soy_masks[year] = (np.asarray(cov_clipped) == SOJA_CODE)

        if ref_coords is None:
            ref_coords = (np.asarray(cov_clipped.y), np.asarray(cov_clipped.x))

        n = int(soy_masks[year].sum())
        log.info(f"  {year}: {n} pixels de soja")

    if ref_coords is None:
        log.error("Sem coordenadas de referência")
        return None

    y_arr, x_arr = ref_coords

    # União de pixels soja em qualquer ano
    union_mask: np.ndarray = np.zeros(soy_masks[years[0]].shape, dtype=bool)
    for m in soy_masks.values():
        # mascara pode ter shape levemente diferente se o clip diferiu (raro)
        h, w = min(union_mask.shape[0], m.shape[0]), min(union_mask.shape[1], m.shape[1])
        union_mask[:h, :w] |= m[:h, :w]

    n_pixels_total = int(union_mask.sum())
    if n_pixels_total == 0:
        log.warning("Nenhum pixel de soja encontrado neste hexágono.")
        return None

    log.info(f"Total de pixels soja (união): {n_pixels_total}")

    # Índices (yi, xi) dos pixels soja
    soy_yi, soy_xi = np.where(union_mask)
    pixel_lons = x_arr[soy_xi]
    pixel_lats = y_arr[soy_yi]

    # ── Passo 2: Baixa VIIRS e extrai EVI por pixel ────────────────────────
    all_rows: list[dict] = []

    for year in years:
        log.info(f"Baixando VIIRS {year}...")
        with tempfile.TemporaryDirectory(prefix="fenologia_px_") as tmp:
            cube = viirs.build_tile_cube(
                list(tiles_hv), year, template, Path(tmp), label
            )

        if cube is None:
            log.warning(f"  {year}: sem dados VIIRS — pulando")
            continue

        # Mantém apenas datas do ano corrente
        cube = cube.sel(time=cube["time"].dt.year == year)
        if cube.sizes["time"] == 0:
            continue

        # Recorta ao hexágono
        cube_clipped = _clip_box_hex(cube, geom)
        dates_pd = pd.to_datetime(np.asarray(cube_clipped.time))

        evi_arr = np.asarray(cube_clipped)  # (time, y, x)

        # Alinha as dimensões do mask com o cubo recortado
        cube_y = np.asarray(cube_clipped.y)
        cube_x = np.asarray(cube_clipped.x)

        for pid, (lon, lat) in enumerate(zip(pixel_lons, pixel_lats)):
            # Encontra o índice mais próximo no cubo recortado
            yi_c = int(np.argmin(np.abs(cube_y - lat)))
            xi_c = int(np.argmin(np.abs(cube_x - lon)))

            # Verifica se está dentro dos limites
            if yi_c >= evi_arr.shape[1] or xi_c >= evi_arr.shape[2]:
                continue

            was_soja = bool(
                soy_masks[year][soy_yi[pid], soy_xi[pid]]
                if soy_yi[pid] < soy_masks[year].shape[0]
                and soy_xi[pid] < soy_masks[year].shape[1]
                else False
            )

            for date, evi_val in zip(dates_pd, evi_arr[:, yi_c, xi_c]):
                evi_f = float(evi_val)
                if np.isnan(evi_f):
                    continue
                all_rows.append({
                    "lon":       round(float(lon), 5),
                    "lat":       round(float(lat), 5),
                    "data":      date,
                    "evi":       round(evi_f, 4),
                    "soja_ano":  was_soja,
                })

        log.info(f"  {year}: {len(all_rows):,} linhas acumuladas")

    if not all_rows:
        log.warning("Nenhuma linha gerada — sem dados EVI para os pixels soja.")
        return None

    df = pd.DataFrame(all_rows)
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values(["lon", "lat", "data"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{hex_id}.parquet"
    df.to_parquet(out_path, index=False)

    # ── Resumo final ───────────────────────────────────────────────────────
    n_px      = df.groupby(["lon", "lat"]).ngroups
    n_obs     = len(df)
    n_soja    = df["soja_ano"].sum()
    evi_stats = df["evi"].describe(percentiles=[.25, .50, .75, .90])
    px_by_year = (
        df[df["soja_ano"]]
        .assign(year=df["data"].dt.year)
        .groupby("year")[["lon","lat"]]
        .apply(lambda g: g.drop_duplicates().shape[0])
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RESULTADO — extração pixel-level")
    print(f"  Hexágono : {hex_id}")
    print(f"  Saída    : {out_path}")
    print(sep)
    print(f"  Pixels únicos (soja ∪ todos os anos) : {n_px:>6,}")
    print(f"  Observações totais                   : {n_obs:>6,}")
    print(f"  Observações com soja_ano=True        : {n_soja:>6,}  ({100*n_soja/max(n_obs,1):.1f}%)")
    print(f"  Período   : {str(df['data'].min())[:10]}  →  {str(df['data'].max())[:10]}")
    print(f"\n  EVI (todos os pixels / datas):")
    print(f"    mín={evi_stats['min']:.4f}  P25={evi_stats['25%']:.4f}  "
          f"med={evi_stats['50%']:.4f}  P75={evi_stats['75%']:.4f}  "
          f"P90={evi_stats['90%']:.4f}  máx={evi_stats['max']:.4f}")
    print(f"\n  Pixels soja por ano (MapBiomas):")
    for year, cnt in px_by_year.items():
        bar = "█" * int(cnt / max(px_by_year.max(), 1) * 30)
        print(f"    {year}: {cnt:>4}  {bar}")
    print(f"\n  Para iniciar o dashboard:")
    print(f"    PIXEL_HEX={hex_id} uvicorn pixel_dashboard.app:app --port 8001")
    print(sep + "\n")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Extrai série EVI pixel-level para um hexágono H3 (somente pixels soja)."
    )
    p.add_argument("hex_id", help="ID do hexágono H3, ex: 85815527fffffff")
    p.add_argument("--years", nargs="+", type=int, default=list(config.YEARS),
                   help=f"Anos a processar (padrão {config.YEARS})")
    p.add_argument("--output", default=str(OUTPUT_DIR),
                   help=f"Pasta de saída (padrão {OUTPUT_DIR})")
    args = p.parse_args()

    viirs.authenticate()
    extract_pixel_series(args.hex_id, args.years, Path(args.output))


if __name__ == "__main__":
    main()
