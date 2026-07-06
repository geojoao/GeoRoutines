"""
Configurações e constantes da rotina de produção (viirs_evi).

Tudo que é "parâmetro de negócio" (classes, produto VIIRS, resolução do grid,
janela temporal, caminhos, blob) fica centralizado aqui. A maioria pode ser
sobrescrita por variável de ambiente para facilitar o deploy via kbatch.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos (disco efêmero do pod)
# ---------------------------------------------------------------------------
# Raiz de trabalho local (dentro do pod). Tudo é temporário — o que precisa
# sobreviver é enviado ao blob.
WORK_DIR = Path(os.environ.get("VIIRS_WORK_DIR", ".")).resolve()

# Pasta onde os GeoTIFFs do MapBiomas são baixados no início do run.
MAPBIOMAS_DIR = Path(os.environ.get("MAPBIOMAS_DIR", WORK_DIR / "mapbiomas"))

# Saída local (parquet canônico + partes por balde).
OUTPUT_DIR = Path(os.environ.get("VIIRS_OUTPUT_DIR", WORK_DIR / "output"))

# Nome do parquet canônico (fonte de verdade do estado incremental).
OUTPUT_PARQUET = os.environ.get("VIIRS_OUTPUT_PARQUET", "evi_brazil.parquet")

# Contorno do Brasil (vetor) usado para gerar o grid H3. Baixado do blob de
# input no início do run (ver run.py). Também aceita uma lista de hexágonos.
BRAZIL_BOUNDARY = Path(os.environ.get("BRAZIL_BOUNDARY", WORK_DIR / "brazil.geojson"))

# ---------------------------------------------------------------------------
# Janela temporal e grid
# ---------------------------------------------------------------------------
# Início da série. VNP13A1 v002 começa em 2012-01-17. Configurável para reduzir
# o custo do primeiro processamento (ex.: "2020-01-01").
VIIRS_START_DATE = os.environ.get("VIIRS_START_DATE", "2012-01-17")

# Fim da série: por padrão, o dia da execução (todos os dados disponíveis).
# Pode ser fixado via env para reprocessamentos determinísticos.
VIIRS_END_DATE = os.environ.get("VIIRS_END_DATE", "").strip() or None

H3_RESOLUTION = int(os.environ.get("VIIRS_H3_RESOLUTION", "5"))

# Resolução do grid regular EPSG:4326 para onde o VIIRS é reprojetado.
# VNP13A1 ~ 463 m  ->  463 / 111320 ~ 0.00416 graus.
TARGET_RES_DEG = 1.0 / 240.0  # ~0.004167 graus (~463 m)

# Buffer (graus) aplicado ao bounding box do hexágono ao recortar/reprojetar.
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
# Acesso aos granules: S3 direto x HTTPS
# ---------------------------------------------------------------------------
# "auto" (S3 direto, fallback HTTPS), "s3" (força S3) ou "https".
# In-region (us-west-2) recomenda-se "s3" para garantir que nada saia por HTTPS.
VIIRS_ACCESS_MODE = os.environ.get("FENOLOGIA_ACCESS", "auto").strip().lower()
VIIRS_S3_PROVIDER = os.environ.get("FENOLOGIA_S3_PROVIDER", "LPCLOUD").strip()
VIIRS_DL_WORKERS = max(1, int(os.environ.get("FENOLOGIA_DL_WORKERS", "4")))
VIIRS_PREFETCH_DATES = max(1, int(os.environ.get("FENOLOGIA_PREFETCH_DATES", "2")))

# ---------------------------------------------------------------------------
# Classes do MapBiomas
# ---------------------------------------------------------------------------
# Macroclasse "Agricultura" (filho de "Agropecuária") na legenda do MapBiomas.
COVERAGE_AGRI_CLASSES = {
    18: "agricultura",
    19: "lavoura_temporaria",
    36: "lavoura_perene",
    39: "soja",
    20: "cana",
    40: "arroz",
    62: "algodao",
    41: "outras_lavouras_temporarias",
    46: "cafe",
    47: "citrus",
    35: "dende",
    48: "outras_lavouras_perenes",
}

# Raster de safrinha (agriculture_agricultural_use_second_crop).
SECOND_CROP_CLASSES = {
    1: "segunda_safra",
    41: "segunda_safra_outras_temporarias",
    62: "segunda_safra_algodao",
}

# nodata dos rasters MapBiomas.
MAPBIOMAS_NODATA = 0

# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------
# Container de input (boundary/lista de hexágonos) e output (parquet).
BLOB_INPUT_CONTAINER = os.environ.get("VIIRS_BLOB_INPUT_CONTAINER", "planetary-routines-input")
BLOB_OUTPUT_CONTAINER = os.environ.get("VIIRS_BLOB_OUTPUT_CONTAINER", "planetary-routines-output")
# Prefixo (namespace) dos objetos de saída no container de output.
BLOB_OUTPUT_PREFIX = os.environ.get("VIIRS_BLOB_OUTPUT_PREFIX", "VIIRS_EVI_HEXAGONO")


def today() -> dt.date:
    return dt.date.today()


def end_date() -> dt.date:
    """Fim da janela: env VIIRS_END_DATE ou o dia da execução."""
    if VIIRS_END_DATE:
        return dt.date.fromisoformat(VIIRS_END_DATE)
    return today()


def start_date() -> dt.date:
    return dt.date.fromisoformat(VIIRS_START_DATE)
