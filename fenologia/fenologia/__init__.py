"""
fenologia_brazil
================

Rotina de engenharia de dados geoespaciais para extrair séries temporais de
EVI (VIIRS / VNP13A1) por classe da macroclasse *Agricultura* do MapBiomas,
agregadas por hexágono H3 (resolução 5) em todo o território brasileiro.

Fluxo geral (ver ``fenologia.pipeline``):

1. Gera o grid de hexágonos H3 res. 5 sobre o Brasil (``fenologia.grid``).
2. Para cada hexágono e cada ano (2020-2024):
   a. Busca/baixa os granules VIIRS VNP13A1 que interceptam o hexágono,
      agrupa por data de composição, faz o mosaico dos tiles do mesmo dia e
      reprojeta para um grid regular em EPSG:4326 (``fenologia.viirs``).
   b. Abre o MapBiomas daquele ano (cobertura geral + safrinha), recorta para
      a região do hexágono e reamostra (maioria) para o mesmo grid do VIIRS
      (``fenologia.mapbiomas``).
   c. Para cada classe da macroclasse agricultura, extrai a série temporal de
      EVI média / mínima / p25 / p75 (``fenologia.extract``).
3. Concatena tudo em um DataFrame (formato *long*) e salva em parquet.
"""

from . import config  # noqa: F401

__all__ = ["config", "grid", "viirs", "mapbiomas", "extract", "pipeline"]
