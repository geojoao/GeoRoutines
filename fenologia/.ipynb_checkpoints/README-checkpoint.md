# fenologia_brazil

Rotina de engenharia de dados geoespaciais que extrai **séries temporais de EVI
do VIIRS** (produto **VNP13A1**, 500 m, composição de 16 dias com passo de 8
dias) para cada **classe da macroclasse _Agricultura_ do MapBiomas**, agregadas
por **hexágono H3 na resolução 5** em todo o território brasileiro, para os anos
de **2020 a 2024**.

Para cada `(hexágono, ano, classe)` o resultado traz a série temporal de
**EVI médio, mínimo, p25 e p75**.

## O que o pipeline faz

A chave da performance é um **lattice global** (bordas de pixel em múltiplos de
`TARGET_RES_DEG`) e **caches por tile VIIRS**, de modo que a reprojeção/
reamostragem cara aconteça **uma vez por tile** e seja reaproveitada por todos
os ~milhares de hexágonos que caem nele (`fenologia/tiles.py`).

**Etapa `prepare` (pesada, uma vez por tile — é onde acontece TODA a rede):**
- **VIIRS** (`fenologia/viirs.py`): por `(tile, ano)`, busca no CMR + baixa os
  `.h5` (com **retry** em erros transitórios do servidor) e reprojeta da grade
  senoidal → EPSG:4326, salvando um COG por `(tile, data)`. Um marcador
  `.prepared_{ano}` evita refazer busca/download.
- **MapBiomas** (`fenologia/mapbiomas.py`): cada raster anual é **reamostrado por
  maioria** (`Resampling.mode`, via `WarpedVRT` em streaming) ao grid do tile e
  salvo como GeoTIFF por `(tile, ano, tipo)`.

Ambos rodam com progress bar antes do loop (controlável com `--no-prepare`).

**Etapa por hexágono (barata, local, sem rede):** apenas **leituras por janela**
dos COGs cacheados — para cada data: recorta os tiles que tocam o hexágono, faz
o **mosaico** e alinha ao grid (sem reprojeção senoidal). Depois, para cada
classe de agricultura, mascara o cubo de EVI e agrega **no espaço** → média /
mínimo / p25 / p75 por data (`fenologia/extract.py`).

Saída: um **parquet por hexágono** no formato *wide* (uma linha por data).

> Caches em `data/viirs_cache` (.h5 brutos), `data/tile_cache/viirs` (EVI por
> tile/data) e `data/tile_cache/mapbiomas` (MapBiomas por tile/ano/tipo).

### As duas fontes do MapBiomas por ano

| Padrão de arquivo | Conteúdo | Classes extraídas |
|---|---|---|
| `{ano}_coverage_lclu*.tif` | Cobertura/uso anual (legenda completa) | Macroclasse Agricultura: soja(39), cana(20), arroz(40), algodão(62), outras lavouras temporárias(41), café(46), citrus(47), dendê(35), outras lavouras perenes(48) + pais 18/19/36 |
| `{ano}_agriculture_agricultural_use_second_crop*.tif` | Tipo da **segunda safra (safrinha)** | Códigos observados: 1, 41, 62 (todos != 0 são segunda safra) |

As classes ficam em `fenologia/config.py` (`COVERAGE_AGRI_CLASSES` e
`SECOND_CROP_CLASSES`) e podem ser ajustadas.

## Esquema do DataFrame de saída (formato *wide*)

Uma linha por `(id_hexagono, data)`, sendo `data` uma **data de composição
nativa do VIIRS** (passo de 8 dias) — **não há nenhuma agregação temporal**. As
estatísticas são **espaciais por classe** (média/min/p25/p75 dos pixels da
classe naquele hexágono e naquela data).

| coluna | descrição |
|---|---|
| `id_hexagono` | ID H3 (res. 5) |
| `data` | data de composição VIIRS (datetime) |
| `evi_medio_<classe>` | média espacial do EVI da classe |
| `evi_min_<classe>` | mínimo espacial do EVI da classe |
| `evi_p25_<classe>` | percentil 25 espacial do EVI da classe |
| `evi_p75_<classe>` | percentil 75 espacial do EVI da classe |

Ex.: `evi_medio_soja`, `evi_min_soja`, `evi_p25_soja`, `evi_p75_soja`,
`evi_medio_algodao`, …, `evi_medio_segunda_safra`, … O schema é **fixo** (todas
as classes de `config.py` viram colunas; classes ausentes no hexágono ficam
NaN), o que mantém todos os parquets com as mesmas colunas. Para inspecionar a
versão *long* (com `classe_codigo`, `n_pixels`, `area_ha`), use
`fenologia.extract.extract_class_series` diretamente.

## Instalação (uv)

```bash
uv sync          # cria o .venv e instala tudo a partir do pyproject/uv.lock
```

Também há um `requirements.txt` exportado para ambientes sem uv
(`pip install -r requirements.txt`).

## Credenciais NASA Earthdata

A **busca** no CMR é anônima, mas o **download** dos granules exige login no
[NASA Earthdata](https://urs.earthdata.nasa.gov/). Configure de uma destas
formas (a autenticação tenta nesta ordem):

```bash
# 1) variáveis de ambiente
export EARTHDATA_USERNAME="seu_usuario"
export EARTHDATA_PASSWORD="sua_senha"

# 2) ou ~/.netrc
machine urs.earthdata.nasa.gov login seu_usuario password sua_senha
```

> Lembre de autorizar a aplicação **"LP DAAC Data Pool"** no seu perfil Earthdata.

## Acesso ao VIIRS: HTTPS x S3 direto

Os granules VIIRS podem ser obtidos por **HTTPS** (padrão fora da AWS) ou
**direto do S3 da LP DAAC** (`us-west-2`). Rodando a rotina **dentro de
`us-west-2`** (ex.: um notebook/instância nessa região), o acesso S3 é
*in-region*: baixa latência e sem custo de egress. Controle pela variável
`VIIRS_ACCESS_MODE`:

| valor | comportamento |
|---|---|
| `auto` (padrão) | usa S3 quando detecta execução in-region (`AWS_REGION`/`AWS_DEFAULT_REGION` == `us-west-2`, ou a detecção do earthaccess); caso contrário, HTTPS |
| `s3` | força o acesso direto ao S3 (use ao rodar em us-west-2) |
| `download` | força o download via HTTPS para o cache local (`.h5`) |

```bash
# Rodando num notebook/instância em us-west-2:
export VIIRS_ACCESS_MODE=s3
uv run python run.py --boundary data/brazil.geojson
```

No modo S3 o granule é **copiado do S3 para o cache local** (`data/viirs_cache`)
e lido com h5py — mesmo footprint de disco do modo HTTPS, só trocando o
transporte. Para ler direto do objeto S3 sem cópia local, use `VIIRS_S3_STREAM=1`
(pode ser mais lento por causa das leituras HDF5 sobre fsspec). A autenticação
Earthdata é a mesma; as credenciais temporárias do S3 são obtidas pelo
earthaccess a partir do seu login. **Todo o resto (MapBiomas, cache por tile,
parquets) continua em disco local, sem mudanças.**

## Uso

```bash
# Região de teste (bbox) — só 2 hexágonos:
uv run python run.py --bbox -48.2 -16.1 -47.8 -15.7 --limit 2

# Lista explícita de hexágonos:
uv run python run.py --hex 858b8633fffffff

# Brasil inteiro (precisa de um contorno do país em data/brazil.geojson):
uv run python run.py --boundary data/brazil.geojson
```

Como biblioteca:

```python
from fenologia.pipeline import process_hexagon
df = process_hexagon("858b8633fffffff", years=[2024])
```

Os parquets são salvos em `data/output/evi_{hexagono}.parquet`. O modo `resume`
(padrão) pula hexágonos já processados.

### Grid H3 do Brasil

`run.py --boundary <arquivo>` gera o grid res. 5 sobre o contorno informado
(GeoJSON/Shapefile/GPKG do Brasil — ex.: IBGE, `geobr`, Natural Earth). São
~33 mil hexágonos cobrindo o país.

## Teste offline (sem NASA)

Valida o leitor de HDF-EOS5, mosaico/reprojeção, reamostragem do MapBiomas e a
extração usando um granule VNP13A1 **sintético** + os rasters MapBiomas reais:

```bash
uv run python tests/test_offline.py
```

## Decisões de projeto

- **Produto VIIRS = VNP13A1 v002** (500 m, EVI). Para trocar por 1 km (VNP13A2)
  ou CMG (VNP13C1), ajuste `VIIRS_SHORT_NAME` em `config.py`.
- **Leitura via `h5py`** (não via driver HDF5 do GDAL, que não vem nos wheels do
  rasterio): o array de EVI é lido e a georreferência senoidal é reconstruída do
  `StructMetadata.0`, depois reprojetada para EPSG:4326.
- **`rioxarray` no lugar do `salem`**: `reproject_match(resampling=mode)` faz a
  reamostragem por maioria (equivalente ao `lookup_transform`) e `rio.clip` faz
  o recorte ao hexágono — sem a dependência frágil do salem.
- **Macroclasse Agricultura**: exclui pastagem (15), silvicultura (9) e mosaico
  de usos (21), que são irmãos mas estão fora da macroclasse Agricultura.
- **Cadência temporal**: o VNP13A1 tem passo de 8 dias (~46 datas/ano). A janela
  por ano inclui as composições de fim do ano anterior que se sobrepõem a
  janeiro, preservando a continuidade fenológica.

## Estrutura

```
fenologia/
  config.py       # constantes (anos, classes, produto VIIRS, caminhos, caches)
  grid.py         # grid H3 res. 5 (h3 v4)
  tiles.py        # matemática dos tiles senoidais + grid alvo por tile (lattice global)
  rasterutils.py  # template do grid alvo (snap ao lattice global)
  viirs.py        # busca/download/leitura HDF5 + cache COG por (tile,data) + cubo
  mapbiomas.py    # reamostragem (maioria) por (tile,ano,tipo) via WarpedVRT + cache
  extract.py      # estatísticas de EVI por classe + pivot p/ formato wide
  pipeline.py     # orquestração (prepare / process_hexagon / run) + progress bar
run.py            # CLI
tests/            # teste offline + gerador de granule sintético
```

## Escala / produção

São ~33 mil hexágonos × 5 anos × ~46 datas — carga pesada. O ganho vem de pagar
a reprojeção/reamostragem **por tile** (não por hexágono):

| operação cara | sem cache | com cache por tile |
|---|---|---|
| reprojeção do tile VIIRS | `N_hex × N_datas × N_anos` | `N_tiles × N_datas` |
| reamostragem MapBiomas (moda) | `N_hex × N_anos × 2` | `N_tiles × N_anos × 2` |

Como há **milhares de hexágonos por tile** (~10°), as operações pesadas caem
~1000×. O trabalho por hexágono vira leitura por janela + agregação.

Recomendações:
- rode a etapa `prepare` (padrão) para materializar os COGs por tile uma vez;
- o modo `resume` permite retomar; paralelize por lote de hexágonos
  (multiprocessing/dask) chamando `process_hexagon` em workers distintos —
  hexágonos do mesmo tile compartilham o cache em disco.
```
