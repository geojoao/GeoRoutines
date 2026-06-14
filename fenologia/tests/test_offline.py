"""
Teste offline ponta-a-ponta (sem credenciais NASA), com a arquitetura por TILES.

Valida:
1. Leitor de HDF-EOS5 senoidal (read_evi_tile) + georreferência.
2. Cache por (tile, data) do VIIRS + montagem do cubo (build_daily_cube com
   search/download monkeypatched para devolver granules sintéticos).
3. Cache por (tile, ano, tipo) do MapBiomas via WarpedVRT (mode) + leitura por
   janela.
4. Extração das séries de EVI por classe no formato WIDE (process_hexagon).

Para manter o teste rápido, ``tiles.tile_grid`` é encurtado (monkeypatch) para a
região do hexágono — assim o WarpedVRT do MapBiomas reamostra só um pedacinho,
mas todo o caminho de cache é exercitado. O granule sintético usa o EXTENT REAL
do tile (h12v10).
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fenologia import config  # noqa: E402

# cache de teste isolado
config.TILE_CACHE_DIR = ROOT / "data" / "_test_tile_cache"
shutil.rmtree(config.TILE_CACHE_DIR, ignore_errors=True)

from fenologia import extract, mapbiomas, pipeline, tiles, viirs  # noqa: E402
from fenologia.grid import cell_to_polygon  # noqa: E402
from fenologia.rasterutils import make_target_grid  # noqa: E402
from tests.make_synthetic_granule import make_synthetic_vnp13a1  # noqa: E402

HEX = "858b8633fffffff"  # Sorriso-MT (soja/algodão/outras temporárias)
TMP = ROOT / "data" / "_test_cache"
H, V = 12, 10  # tile VIIRS que contém o hexágono

# Encurta tile_grid para a região do hexágono (rápido, mas exercita o cache).
_geom = cell_to_polygon(HEX)
_b = config.BBOX_BUFFER_DEG
_minx, _miny, _maxx, _maxy = _geom.bounds
_SMALL_GRID = make_target_grid((_minx - _b, _miny - _b, _maxx + _b, _maxy + _b))
tiles.tile_grid = lambda h, v, res=config.TARGET_RES_DEG: _SMALL_GRID


def _template():
    return make_target_grid((_minx - _b, _miny - _b, _maxx + _b, _maxy + _b))


def _fake_granules(prefix="t"):
    base = dt.date(2024, 1, 1)
    dates = [base + dt.timedelta(days=16 * k) for k in range(6)]
    seasons = [0.4, 0.6, 0.8, 0.95, 0.7, 0.5]
    grans, paths = [], {}
    for k, (d, s) in enumerate(zip(dates, seasons)):
        ur = f"VNP13A1.A2024{d.timetuple().tm_yday:03d}.h{H:02d}v{V:02d}.002.{prefix}{k}"
        grans.append({"umm": {"GranuleUR": ur}})
        paths[ur] = make_synthetic_vnp13a1(TMP / f"{ur}.h5", H, V, season=s, seed=k)
    return grans, paths


def test_read_tile_georef():
    p = make_synthetic_vnp13a1(TMP / "tile0.h5", H, V, season=0.8)
    da = viirs.read_evi_tile(p)
    assert da.rio.crs is not None
    assert np.isnan(np.asarray(da)[0, 0])  # fill -> NaN
    print(f"[1] read_evi_tile OK | shape={tuple(da.shape)} "
          f"EVI[min,max]=[{np.nanmin(da.values):.3f},{np.nanmax(da.values):.3f}]")


def test_tiles_for_geometry():
    hv = tiles.tiles_for_geometry(_geom)
    assert (H, V) in hv, f"esperava ({H},{V}) em {hv}"
    print(f"[2] tiles_for_geometry OK | tiles={hv}")


def test_build_cube_cached():
    grans, paths = _fake_granules("c")
    template = _template()
    os_, od = viirs.search_granules_bbox, viirs.download_granules
    viirs.search_granules_bbox = lambda bbox, start, end: list(grans)
    viirs.download_granules = lambda granules, cache_dir=None: dict(paths)
    try:
        cube = viirs.build_daily_cube(_geom, "2024-01-01", "2024-12-31", template)
    finally:
        viirs.search_granules_bbox, viirs.download_granules = os_, od

    assert cube is not None and cube.dims == ("time", "y", "x")
    assert cube.sizes["time"] == 6
    # COGs por (tile,data) materializados em cache
    cogs = sorted((config.TILE_CACHE_DIR / "viirs" / f"h{H:02d}v{V:02d}").glob("*.tif"))
    assert len(cogs) == 6, f"esperava 6 COGs VIIRS, achei {len(cogs)}"
    means = [float(np.nanmean(cube.isel(time=t).values)) for t in range(6)]
    assert means[3] == max(means)
    print(f"[3] build_daily_cube (cache) OK | grid={cube.sizes['y']}x{cube.sizes['x']} "
          f"cogs={len(cogs)} medias={[round(m,3) for m in means]}")


def test_mapbiomas_cache():
    template = _template()
    cov = mapbiomas.read_mapbiomas_on_grid(2024, "coverage", template, [(H, V)])
    sec = mapbiomas.read_mapbiomas_on_grid(2024, "second_crop", template, [(H, V)])
    assert cov.sizes == template.sizes
    cog = mapbiomas.mapbiomas_tile_cog_path(H, V, 2024, "coverage")
    assert cog.exists(), "COG do MapBiomas não foi cacheado"
    cov_classes = sorted(set(np.unique(cov.values).tolist()))
    assert set(config.COVERAGE_AGRI_CLASSES) & set(cov_classes)
    print(f"[4] mapbiomas cache OK | cobertura={cov_classes} "
          f"safrinha={sorted(set(np.unique(sec.values).tolist()))}")


def test_full_pipeline_wide():
    grans, paths = _fake_granules("f")
    os_, od = viirs.search_granules_bbox, viirs.download_granules
    viirs.search_granules_bbox = lambda bbox, start, end: list(grans)
    viirs.download_granules = lambda granules, cache_dir=None: dict(paths)
    try:
        df = pipeline.process_hexagon(HEX, years=[2024])
    finally:
        viirs.search_granules_bbox, viirs.download_granules = os_, od

    assert len(df) > 0
    assert {"id_hexagono", "data"} <= set(df.columns)
    assert df["data"].is_unique and df["data"].nunique() == 6
    assert list(df.columns) == extract.full_wide_columns()
    for stat in ("evi_medio", "evi_min", "evi_p25", "evi_p75"):
        assert f"{stat}_soja" in df.columns

    for name in ("soja", "outras_lavouras_temporarias"):
        sub = df.dropna(subset=[f"evi_medio_{name}"])
        if len(sub):
            assert (sub[f"evi_min_{name}"] <= sub[f"evi_p25_{name}"] + 1e-6).all()
            assert (sub[f"evi_p25_{name}"] <= sub[f"evi_medio_{name}"] + 1e-6).all()
            assert (sub[f"evi_medio_{name}"] <= sub[f"evi_p75_{name}"] + 1e-6).all()

    classes = sorted(
        c.replace("evi_medio_", "")
        for c in df.columns
        if c.startswith("evi_medio_") and df[c].notna().any()
    )
    print(f"[5] process_hexagon (WIDE) OK | linhas={len(df)} datas={df['data'].nunique()} "
          f"colunas={len(df.columns)} classes_com_dado={classes}")
    show = ["id_hexagono", "data"]
    for name in classes:
        show += [f"evi_medio_{name}", f"evi_min_{name}", f"evi_p25_{name}", f"evi_p75_{name}"]
    print(df[show].head(6).to_string(index=False))
    return df


if __name__ == "__main__":
    test_read_tile_georef()
    test_tiles_for_geometry()
    test_build_cube_cached()
    test_mapbiomas_cache()
    df = test_full_pipeline_wide()
    print("\n=== TODOS OS TESTES OFFLINE PASSARAM ===")
