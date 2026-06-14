"""
Cria um granule VNP13A1 SINTÉTICO (HDF-EOS5) com a mesma estrutura do produto
real (grade senoidal + StructMetadata + Data Field de EVI), no EXTENT REAL de um
tile (h, v). Usado só para testes offline (sem acesso à NASA).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fenologia import config
from fenologia.tiles import tile_sinu_bounds


def make_synthetic_vnp13a1(
    path: Path,
    h: int,
    v: int,
    n: int = 1200,
    season: float = 1.0,
    seed: int = 0,
) -> Path:
    """Gera um .h5 (tile senoidal real h/v, n x n) com EVI sintético."""
    ulx, uly, lrx, lry = tile_sinu_bounds(h, v)

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    base = 0.25 + 0.6 * (xx / n) + 0.05 * rng.standard_normal((n, n))
    base = np.clip(base, 0.0, 0.95)
    evi_phys = np.clip(base * season, -0.2, 1.0)
    stored = (evi_phys / config.VIIRS_EVI_SCALE).astype("int16")  # *10000

    # alguns pixels de fill, para testar o mascaramento
    stored[:5, :5] = config.VIIRS_EVI_FILL

    meta = (
        "GROUP=GridStructure\n"
        "\tGROUP=GRID_1\n"
        '\t\tGridName="VIIRS_Grid_16Day_VI_500m"\n'
        f"\t\tXDim={n}\n"
        f"\t\tYDim={n}\n"
        f"\t\tUpperLeftPointMtrs=({ulx:.6f},{uly:.6f})\n"
        f"\t\tLowerRightMtrs=({lrx:.6f},{lry:.6f})\n"
        "\tEND_GROUP=GRID_1\n"
        "END_GROUP=GridStructure\n"
        "END\n"
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        df_grp = h5.create_group("HDFEOS/GRIDS/VIIRS_Grid_16Day_VI_500m/Data Fields")
        for name, arr in [
            ("500 m 16 days EVI", stored),
            ("500 m 16 days EVI2", stored // 2),  # garante que o finder pega EVI, não EVI2
            ("500 m 16 days NDVI", stored),
        ]:
            d = df_grp.create_dataset(name, data=arr)
            d.attrs["_FillValue"] = np.int16(config.VIIRS_EVI_FILL)
            d.attrs["scale_factor"] = np.float64(config.VIIRS_EVI_SCALE)
            d.attrs["valid_range"] = np.array(config.VIIRS_EVI_VALID, dtype="int16")

        info = h5.create_group("HDFEOS INFORMATION")
        info.create_dataset("StructMetadata.0", data=np.bytes_(meta.encode()))

    return path
