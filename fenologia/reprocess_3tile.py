"""
Reprocessa baldes multi-tile que causam OOM quando o template cobre toda a
extensão dos tiles. Usa um template reduzido ao bounding box dos hexágonos.

Trata os buckets:
  - h13v11+h13v12 (2-tile, 6 hex)
  - h12v11+h13v11+h13v12 (3-tile, 33 hex)
  - h13v11+h13v12+h14v12 (3-tile, 16 hex)
"""
from __future__ import annotations

import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import h3
import pandas as pd
import shapely.geometry as sg
from tqdm import tqdm

from fenologia import config, extract, mapbiomas, tiles, viirs
from fenologia.pipeline import _part_path
from fenologia.grid import cell_to_polygon
from fenologia.rasterutils import make_target_grid
from shapely.geometry import mapping

PARTS_DIR = Path(config.OUTPUT_DIR) / "_parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_FILES = {
    "h13v11+h13v12": "data/hex_h13v11_h13v12.txt",
    "h12v11+h13v11+h13v12": "data/hex_h12v11_h13v11_h13v12.txt",
    "h13v11+h13v12+h14v12": "data/hex_h13v11_h13v12_h14v12.txt",
}

TILE_MAP = {
    "h13v11+h13v12": [(13, 11), (13, 12)],
    "h12v11+h13v11+h13v12": [(12, 11), (13, 11), (13, 12)],
    "h13v11+h13v12+h14v12": [(13, 11), (13, 12), (14, 12)],
}


def hex_bbox(hex_ids: list[str], buffer: float = 0.05) -> tuple[float, float, float, float]:
    bounds = []
    for h in hex_ids:
        boundary = h3.cell_to_boundary(h)
        coords = [(lon, lat) for lat, lon in boundary]
        geom = sg.Polygon(coords)
        bounds.append(geom.bounds)
    minx = min(b[0] for b in bounds) - buffer
    miny = min(b[1] for b in bounds) - buffer
    maxx = max(b[2] for b in bounds) + buffer
    maxy = max(b[3] for b in bounds) + buffer
    return minx, miny, maxx, maxy


def _clip_to_hex(da, geom):
    return da.rio.clip([mapping(geom)], crs="EPSG:4326", drop=False, all_touched=True)


def _hex_window(hex_id: str) -> tuple[float, float, float, float]:
    geom = cell_to_polygon(hex_id)
    minx, miny, maxx, maxy = geom.bounds
    b = config.BBOX_BUFFER_DEG
    return minx - b, miny - b, maxx + b, maxy + b


def process_bucket(label: str, hex_ids: list[str], tiles_hv: list[tuple[int, int]]) -> None:
    part_path = PARTS_DIR / f"balde_{label}.parquet"

    # Hex-bounded template: much smaller than the full tile extent
    bbox = hex_bbox(hex_ids)
    template = make_target_grid(bbox, config.TARGET_RES_DEG)
    print(f"  Template: {template.sizes['x']}×{template.sizes['y']} pixels "
          f"(bbox {bbox[0]:.2f},{bbox[1]:.2f} → {bbox[2]:.2f},{bbox[3]:.2f})")

    frames_by_hex: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for year in config.YEARS:
        with tempfile.TemporaryDirectory(prefix="fenologia_") as tmp:
            cube = viirs.build_tile_cube(tiles_hv, year, template, Path(tmp), f"{label} {year}")
        if cube is None:
            print(f"  [{label} {year}] nenhum granule encontrado")
            continue

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
                hex_id=hex_id, year=year, source="coverage",
                res_deg=config.TARGET_RES_DEG,
            )
            df_sec = extract.extract_class_series(
                cube_hex, sec_hex, config.SECOND_CROP_CLASSES,
                hex_id=hex_id, year=year, source="second_crop",
                res_deg=config.TARGET_RES_DEG,
            )
            if len(df_cov):
                frames_by_hex[hex_id].append(df_cov)
            if len(df_sec):
                frames_by_hex[hex_id].append(df_sec)

        print(f"  [{label} {year}] cube processado, iniciando próximo ano")

    result_frames = []
    for hex_id in hex_ids:
        frames = frames_by_hex.get(hex_id)
        if frames:
            result_frames.append(extract.to_wide(pd.concat(frames, ignore_index=True)))
        else:
            result_frames.append(extract.to_wide(None))

    bucket_df = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame(columns=extract.full_wide_columns())
    )
    bucket_df.to_parquet(part_path, index=False)
    n_with_data = sum(1 for f in result_frames if len(f) > 0)
    print(f"  [{label}] Salvo {part_path.name}: {n_with_data}/{len(hex_ids)} hex com dados")


def main():
    for label, hex_file in BUCKET_FILES.items():
        part_path = PARTS_DIR / f"balde_{label}.parquet"
        if part_path.exists():
            print(f"Pulando {label}: parquet já existe ({part_path.stat().st_size/1024:.0f} KB)")
            continue

        with open(hex_file) as f:
            hex_ids = f.read().strip().split()
        tiles_hv = TILE_MAP[label]

        print(f"\nProcessando {label}: {len(hex_ids)} hex, tiles {tiles_hv}")
        t0 = time.time()
        process_bucket(label, hex_ids, tiles_hv)
        print(f"  Tempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
