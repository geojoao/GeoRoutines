"""
Teste offline ponta-a-ponta (sem credenciais NASA), com a arquitetura por
"balde de tiles" (sem cache em disco).

Valida:
1. Leitor de HDF-EOS5 senoidal (read_evi_tile) + georreferência.
2. tiles_for_geometry / bucket_grid.
3. build_tile_cube (busca + abertura de granules + reprojeção, tudo em
   memória, com download/open monkeypatched para granules sintéticos).
4. Leitura do MapBiomas direto via WarpedVRT (sem cache em disco).
5. Extração das séries de EVI por classe no formato WIDE
   (pipeline.process_tile_bucket).

Para manter o teste rápido, ``tiles.tile_latlon_bbox`` é encurtado
(monkeypatch) para a região do hexágono — assim ``bucket_grid``/
``tiles_for_geometry`` operam num grid pequeno, mas todo o caminho de
download->leitura->reprojeção->extração é exercitado. O granule sintético usa
o EXTENT REAL do tile (h12v10).
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fenologia import config  # noqa: E402
from fenologia import extract, mapbiomas, pipeline, tiles, viirs  # noqa: E402
from fenologia.grid import cell_to_polygon  # noqa: E402
from tests.make_synthetic_granule import make_synthetic_vnp13a1  # noqa: E402

HEX = "858b8633fffffff"  # Sorriso-MT (soja/algodão/outras temporárias)
TMP = ROOT / "data" / "_test_cache"
H, V = 12, 10  # tile VIIRS que contém o hexágono

shutil.rmtree(TMP, ignore_errors=True)

# Encurta tile_latlon_bbox para a região do hexágono (rápido, mas exercita o
# fluxo real de bucket_grid/tiles_for_geometry).
_geom = cell_to_polygon(HEX)
_b = config.BBOX_BUFFER_DEG
_minx, _miny, _maxx, _maxy = _geom.bounds
_SMALL_BBOX = (_minx - _b, _miny - _b, _maxx + _b, _maxy + _b)

tiles.tile_latlon_bbox = lambda h, v: _SMALL_BBOX if (h, v) == (H, V) else None


def _bucket_template():
    return tiles.bucket_grid([(H, V)])


class _FakeGranuleFile:
    """Stand-in p/ o file-like devolvido por earthaccess.open (lê de um .h5 local)."""

    def __init__(self, path: Path):
        self._f = open(path, "rb")

    def read(self, *args, **kwargs):
        return self._f.read(*args, **kwargs)

    def close(self):
        self._f.close()


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


def _patch_viirs_network(grans, paths):
    """Monkeypatch search_granules_bbox + open_granule p/ granules sintéticos."""
    orig_search = viirs.search_granules_bbox
    orig_open = viirs.open_granule
    viirs.search_granules_bbox = lambda bbox, start, end: list(grans)
    viirs.open_granule = lambda g: _FakeGranuleFile(paths[viirs.granule_ur(g)])

    def _restore():
        viirs.search_granules_bbox = orig_search
        viirs.open_granule = orig_open

    return _restore


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


def test_build_tile_cube():
    grans, paths = _fake_granules("c")
    template = _bucket_template()
    restore = _patch_viirs_network(grans, paths)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cube = viirs.build_tile_cube([(H, V)], 2024, template, Path(tmp), "test")
    finally:
        restore()

    assert cube is not None and cube.dims == ("time", "y", "x")
    assert cube.sizes["time"] == 6
    means = [float(np.nanmean(cube.isel(time=t).values)) for t in range(6)]
    assert means[3] == max(means)
    print(f"[3] build_tile_cube OK | grid={cube.sizes['y']}x{cube.sizes['x']} "
          f"medias={[round(m, 3) for m in means]}")


def test_mapbiomas_no_cache():
    template = _bucket_template()
    cov = mapbiomas.read_mapbiomas_on_grid(2024, "coverage", template)
    sec = mapbiomas.read_mapbiomas_on_grid(2024, "second_crop", template)
    assert cov.sizes == template.sizes
    cov_classes = sorted(set(np.unique(cov.values).tolist()))
    assert set(config.COVERAGE_AGRI_CLASSES) & set(cov_classes)
    print(f"[4] mapbiomas (sem cache) OK | cobertura={cov_classes} "
          f"safrinha={sorted(set(np.unique(sec.values).tolist()))}")


def test_full_pipeline_wide():
    grans, paths = _fake_granules("f")
    restore = _patch_viirs_network(grans, paths)
    try:
        results = pipeline.process_tile_bucket([HEX], [(H, V)], years=[2024])
    finally:
        restore()

    df = results[HEX]
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
    print(f"[5] process_tile_bucket (WIDE) OK | linhas={len(df)} datas={df['data'].nunique()} "
          f"colunas={len(df.columns)} classes_com_dado={classes}")
    show = ["id_hexagono", "data"]
    for name in classes:
        show += [f"evi_medio_{name}", f"evi_min_{name}", f"evi_p25_{name}", f"evi_p75_{name}"]
    print(df[show].head(6).to_string(index=False))
    return df


if __name__ == "__main__":
    test_read_tile_georef()
    test_tiles_for_geometry()
    test_build_tile_cube()
    test_mapbiomas_no_cache()
    df = test_full_pipeline_wide()
    print("\n=== TODOS OS TESTES OFFLINE PASSARAM ===")
