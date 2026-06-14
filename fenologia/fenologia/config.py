"""
Configurações e constantes do projeto.

Tudo que é "parâmetro de negócio" (classes, anos, produto VIIRS, resolução do
grid alvo, caminhos) fica centralizado aqui para facilitar ajustes.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# Raiz do projeto (pasta que contém este pacote)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Pasta com os GeoTIFFs do MapBiomas já baixados.
MAPBIOMAS_DIR = Path(os.environ.get("MAPBIOMAS_DIR", PROJECT_ROOT / "mapbiomas"))

# Saída dos parquets por hexágono.
OUTPUT_DIR = Path(os.environ.get("FENOLOGIA_OUTPUT_DIR", PROJECT_ROOT / "data" / "output"))

# Boundary do Brasil (vetor) usado para gerar o grid H3. Opcional: se não
# existir, o grid pode ser gerado a partir de um bbox (ver fenologia.grid).
BRAZIL_BOUNDARY = Path(os.environ.get("BRAZIL_BOUNDARY", PROJECT_ROOT / "data" / "brazil.geojson"))

# ---------------------------------------------------------------------------
# Período e grid
# ---------------------------------------------------------------------------
YEARS = list(range(2020, 2025))  # 2020..2024 (inclusive)
H3_RESOLUTION = 5

# Resolução do grid regular EPSG:4326 para onde o VIIRS é reprojetado.
# VNP13A1 ~ 463 m  ->  463 / 111320 ~ 0.00416 graus.
TARGET_RES_DEG = 1.0 / 240.0  # ~0.004167 graus (~463 m)

# Buffer (graus) aplicado ao bounding box do hexágono ao baixar/reprojetar,
# para garantir cobertura total nas bordas.
BBOX_BUFFER_DEG = 0.01

# ---------------------------------------------------------------------------
# Produto VIIRS
# ---------------------------------------------------------------------------
VIIRS_SHORT_NAME = "VNP13A1"   # Vegetation Indices 16-Day L3 Global 500 m
VIIRS_VERSION = "002"
VIIRS_EVI_KEYWORD = "EVI"      # nome do Data Field contém "EVI" (e não "EVI2")

# Valores padrão caso os atributos não estejam presentes no HDF5.
VIIRS_EVI_SCALE = 0.0001       # físico = armazenado * scale
VIIRS_EVI_FILL = -15000
VIIRS_EVI_VALID = (-2000, 10000)

# CRS senoidal do grid VIIRS/MODIS (esfera de raio 6371007.181 m).
VIIRS_SINU_PROJ4 = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
)

# ---------------------------------------------------------------------------
# Classes do MapBiomas
# ---------------------------------------------------------------------------
# Macroclasse "Agricultura" (filho de "Agropecuária") na legenda do MapBiomas.
# Inclui as classes-folha de lavoura temporária e perene + os códigos-pai
# (18/19/36) caso apareçam no raster. NÃO inclui pastagem (15), silvicultura
# (9) nem mosaico de usos (21), que são irmãos mas estão fora da macroclasse
# "Agricultura" propriamente dita.
COVERAGE_AGRI_CLASSES = {
    # pais (raramente presentes nos rasters já desagregados)
    18: "agricultura",
    19: "lavoura_temporaria",
    36: "lavoura_perene",
    # lavoura temporária (folhas)
    39: "soja",
    20: "cana",
    40: "arroz",
    62: "algodao",
    41: "outras_lavouras_temporarias",
    # lavoura perene (folhas)
    46: "cafe",
    47: "citrus",
    35: "dende",
    48: "outras_lavouras_perenes",
}

# Raster de safrinha (agriculture_agricultural_use_second_crop). Codificação
# própria observada nos dados (códigos 1, 41 e 62). Todos os valores != 0 são,
# por definição, segunda safra agrícola.
SECOND_CROP_CLASSES = {
    1: "segunda_safra",
    41: "segunda_safra_outras_temporarias",
    62: "segunda_safra_algodao",
}

# Padrões de nome de arquivo do MapBiomas (resolvidos via glob em MAPBIOMAS_DIR).
MAPBIOMAS_PATTERNS = {
    "coverage": "{year}_coverage_lclu*.tif",
    "second_crop": "{year}_agriculture_agricultural_use_second_crop*.tif",
}

# nodata dos rasters MapBiomas.
MAPBIOMAS_NODATA = 0
