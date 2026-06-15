# fenologia_brazil

Rotina de engenharia de dados geoespaciais que extrai **séries temporais de EVI
do VIIRS** (produto **VNP13A1**, 500 m, composição de 16 dias com passo de 8
dias) para cada **classe da macroclasse _Agricultura_ do MapBiomas**, agregadas
por **hexágono H3 na resolução 5** em todo o território brasileiro, para os anos
de **2020 a 2024**.

Para cada `(hexágono, ano, classe)` o resultado traz a série temporal de
**EVI médio, mínimo, p25 e p75**.

## O que o pipeline faz

**Não há cache em disco.** Os hexágonos H3 são agrupados em "baldes" pelo
conjunto de tiles VIIRS senoidais que cada um toca
(`tiles.tiles_for_geometry`, `fenologia/tiles.py`) — normalmente um balde =
um tile, cobrindo ~milhares de hexágonos vizinhos. Para cada balde e cada ano
(`fenologia.pipeline.process_tile_bucket`):

- **VIIRS** (`fenologia/viirs.py`, `build_tile_cube`): busca no CMR os
  granules `(tile, data)` do ano (com **retry** em erros transitórios), abre
  cada um via `earthaccess.open` — **S3 direto** quando o ambiente está
  in-region na AWS region da LP DAAC, HTTPS caso contrário — copia para um
  `.h5` **temporário**, lê o EVI (h5py) e reprojeta da grade senoidal →
  EPSG:4326 **direto para o grid do balde, em memória**. O `.h5` temporário é
  apagado imediatamente após a leitura. Cada granule aberto gera um log
  `acesso via S3 (...)` ou `acesso via HTTPS (...)` — confira esses logs para
  validar que o tráfego está saindo pelo S3.
- **MapBiomas** (`fenologia/mapbiomas.py`, `read_mapbiomas_on_grid`): lê o
  GeoTIFF anual (COG com overviews) direto via `WarpedVRT`
  (`Resampling.mode`, maioria) para o grid do balde — sem nenhum arquivo
  intermediário; o GDAL só busca os blocos necessários.

O cubo VIIRS e os rasters MapBiomas do balde/ano ficam **em memória** só
durante o processamento desse balde/ano: para cada hexágono do balde, recorta
(`clip_box`/`clip`, barato) à sua janela, mascara o cubo de EVI por classe de
agricultura e agrega **no espaço** → média / mínimo / p25 / p75 por data
(`fenologia/extract.py`). Ao fim do ano, tudo é descartado antes do próximo
balde/ano.

### Saída

Cada balde gera **um parquet** com os dados (formato *wide*, uma linha por
`(id_hexagono, data)`) de **todos** os hexágonos do balde, em
`data/output/_parts/balde_<label>.parquet` (ex.: `balde_h12v08.parquet`,
`balde_h12v09+h13v09.parquet`) — inclusive quando nenhum hexágono do balde tem
agricultura (parquet com 0 linhas), pois esse arquivo também é o **marcador de
balde concluído** usado pelo `resume`. Ao final da execução (ou a qualquer
momento, via `fenologia.pipeline.concat_parts`), todos os `_parts/*.parquet`
são concatenados num único `data/output/evi_brazil.parquet`.

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
**direto do S3 da LP DAAC** (`us-west-2`). O `earthaccess.open()`
(`fenologia/viirs.py`, `open_granule`) escolhe automaticamente: S3 direto
quando o ambiente está *in-region* em `us-west-2` (baixa latência, sem custo
de egress), HTTPS caso contrário — não há nenhuma variável de ambiente para
configurar.

Para cada granule baixado, `build_tile_cube` loga o tipo de acesso usado
(`_access_kind`, que inspeciona a classe real por trás do `EarthAccessFile`):

```
[h12v10 2024-01-01] VNP13A1.A2024001.h12v10.002.xxxx: acesso via S3 (s3fs.core.S3File)
```

ou `acesso via HTTPS (...)` fora de `us-west-2`. Confira essas linhas para
validar que o tráfego está saindo pelo S3. O granule é copiado para um `.h5`
**temporário**, lido com h5py e apagado imediatamente — a autenticação
Earthdata é a mesma em ambos os casos (as credenciais temporárias do S3 são
obtidas pelo earthaccess a partir do seu login).

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
from fenologia.grid import cell_to_polygon
from fenologia.pipeline import process_tile_bucket
from fenologia.tiles import tiles_for_geometry

hex_id = "858b8633fffffff"
tiles_hv = tuple(sorted(tiles_for_geometry(cell_to_polygon(hex_id))))
df = process_tile_bucket([hex_id], tiles_hv, years=[2024])[hex_id]
```

Resultado final em `data/output/evi_brazil.parquet` (todos os hexágonos e
anos, num único parquet). O modo `resume` (padrão) pula **baldes** cujo
`data/output/_parts/balde_<label>.parquet` já existe — um balde só é
(re)processado se ainda não tiver sido concluído (`--no-resume` força
reprocessar tudo).

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
  config.py       # constantes (anos, classes, produto VIIRS, caminhos)
  grid.py         # grid H3 res. 5 (h3 v4)
  tiles.py        # matemática dos tiles senoidais + grid do balde (lattice global)
  rasterutils.py  # template do grid alvo (snap ao lattice global)
  viirs.py        # busca/download (S3 ou HTTPS) + leitura HDF5 + cubo em memória
  mapbiomas.py    # reamostragem (maioria) via WarpedVRT, direto p/ grid do balde
  extract.py      # estatísticas de EVI por classe + pivot p/ formato wide
  pipeline.py     # orquestração (process_tile_bucket / run) + progress bar
run.py            # CLI
tests/            # teste offline + gerador de granule sintético
```

## Escala / produção

São ~33 mil hexágonos × 5 anos × ~46 datas — carga pesada. O ganho vem de
agrupar os hexágonos em **baldes de tiles** (`tiles.tiles_for_geometry`):
para cada balde (normalmente 1 tile ~10°, cobrindo milhares de hexágonos
vizinhos) e cada ano, a reprojeção do EVI VIIRS e a reamostragem do MapBiomas
são feitas **uma única vez em memória** (`process_tile_bucket`) e reusadas
para todos os hexágonos do balde — o trabalho por hexágono vira recorte por
janela + agregação. Ao fim do balde/ano, tudo é descartado; nada fica em
disco entre execuções.

Recomendações:
- o modo `resume` (padrão) pula **baldes** cujo `_parts/balde_<label>.parquet`
  já existe (inclusive baldes sem agricultura, que geram um parquet de 0
  linhas só como marcador) — assim uma execução interrompida não reprocessa
  baldes já concluídos;
- para acompanhar o progresso sem esperar o fim da execução, chame
  `fenologia.pipeline.concat_parts("data/output")` a qualquer momento — gera
  um snapshot de `evi_brazil.parquet` com os baldes já concluídos até então;
- para paralelizar, distribua **baldes inteiros** entre workers
  (multiprocessing/dask) chamando `process_tile_bucket` — assim cada granule
  VIIRS e cada janela do MapBiomas são baixados/lidos uma só vez por
  balde/ano, independente de quantos hexágonos ele contém.
```
