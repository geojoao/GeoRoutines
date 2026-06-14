"""
Geração do grid de hexágonos H3 (resolução 5) sobre o Brasil.

API h3 v4:
- ``h3.LatLngPoly(exterior_latlng, *holes_latlng)`` recebe anéis em (lat, lng).
- ``h3.polygon_to_cells(poly, res)`` preenche o polígono com células.
- ``h3.cell_to_boundary(cell)`` retorna vértices em (lat, lng) -> inverter p/ shapely.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape

from . import config


def cell_to_polygon(cell: str) -> Polygon:
    """Geometria (lon/lat, EPSG:4326) de uma célula H3."""
    # cell_to_boundary devolve (lat, lng); shapely espera (lng, lat).
    coords = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
    return Polygon(coords)


def _polygon_to_latlng_poly(poly: Polygon) -> h3.LatLngPoly:
    exterior = [(lat, lng) for lng, lat in poly.exterior.coords]
    holes = [[(lat, lng) for lng, lat in ring.coords] for ring in poly.interiors]
    return h3.LatLngPoly(exterior, *holes)


def cells_covering_polygon(geom, resolution: int = config.H3_RESOLUTION) -> set[str]:
    """Conjunto de células H3 cujo centro cai dentro de ``geom`` (Polygon/MultiPolygon)."""
    cells: set[str] = set()
    polys: Iterable[Polygon]
    if isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif isinstance(geom, Polygon):
        polys = [geom]
    else:  # shapely geometry collection ou similar
        polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    for poly in polys:
        if poly.is_empty:
            continue
        cells.update(h3.polygon_to_cells(_polygon_to_latlng_poly(poly), resolution))
    return cells


def build_h3_grid(
    boundary: Optional[gpd.GeoDataFrame] = None,
    bbox: Optional[Sequence[float]] = None,
    resolution: int = config.H3_RESOLUTION,
) -> gpd.GeoDataFrame:
    """
    Constrói o GeoDataFrame de hexágonos H3.

    Parâmetros
    ----------
    boundary : GeoDataFrame, opcional
        Limite (ex.: contorno do Brasil). Usa-se a união de todas as geometrias.
    bbox : (minx, miny, maxx, maxy), opcional
        Caixa para gerar o grid (útil para testes em uma região menor).
        Ignorado se ``boundary`` for fornecido.
    resolution : int
        Resolução H3 (padrão 5).

    Retorna
    -------
    GeoDataFrame com colunas ['h3', 'geometry'] em EPSG:4326.
    """
    if boundary is not None:
        geom = boundary.to_crs(4326).union_all()
    elif bbox is not None:
        geom = box(*bbox)
    else:
        raise ValueError("Forneça 'boundary' (GeoDataFrame) ou 'bbox'.")

    cells = sorted(cells_covering_polygon(geom, resolution))
    geometries = [cell_to_polygon(c) for c in cells]
    gdf = gpd.GeoDataFrame({"h3": cells}, geometry=geometries, crs="EPSG:4326")
    return gdf


def load_brazil_boundary(path=None) -> gpd.GeoDataFrame:
    """
    Carrega o contorno do Brasil a partir de um arquivo vetorial.

    Se ``path`` não for informado usa ``config.BRAZIL_BOUNDARY``. Caso o arquivo
    não exista, levanta um erro com instruções (não tenta baixar nada para
    manter o pipeline previsível/offline-friendly).
    """
    path = path or config.BRAZIL_BOUNDARY
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Boundary do Brasil não encontrado em '{path}'.\n"
            "Baixe um contorno do país (ex.: IBGE/geobr/Natural Earth) e salve "
            "nesse caminho, ou passe um GeoDataFrame para build_h3_grid(), ou use "
            "o parâmetro bbox=(minx,miny,maxx,maxy) para uma região de teste."
        )
    return gpd.read_file(path)
