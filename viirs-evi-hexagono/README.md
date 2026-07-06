# viirs-evi-hexagono

Versão de **produção** da rotina de fenologia (`../fenologia`), preparada para
rodar via **kbatch** num pod (disco não-persistente), extraindo séries
temporais de **EVI do VIIRS (VNP13A1, 500 m, composição de 16 dias / passo de 8
dias)** por **hexágono H3 (res. 5)**, mascaradas pela **macroclasse Agricultura
do MapBiomas**, para todo o Brasil.

A pasta `../fenologia` permanece **intocada** — este pacote reaproveita os
módulos-núcleo dela (tiles, grid, extração, leitura HDF5, acesso S3) e adiciona
o que produção precisa: pré-download das máscaras, processamento **incremental**
e I/O via **Azure Blob Storage**.

## O que muda em relação à `fenologia`

1. **Acesso S3 direto ao VIIRS** (in-region us-west-2). Abre o granule pelo link
   `s3://` via `s3fs` com credenciais temporárias do Earthdata, sem depender da
   auto-detecção de região do earthaccess (que falha em pods Kubernetes). No
   deploy use `FENOLOGIA_ACCESS=s3`.

2. **Toda a série disponível**: a janela é `[VIIRS_START_DATE, hoje]`. O VNP13A1
   v002 começa em **2012-01-17** (default). Ajuste `VIIRS_START_DATE` para
   reduzir o custo do primeiro run.

3. **Processamento incremental** (a rotina "entende" o que já existe): lê o
   `evi_brazil.parquet` existente (baixado do blob de saída no início) e, para
   cada hexágono, olha a **última data já gravada**:
   - hexágono **não existe** no parquet → processa **toda** a série disponível;
   - hexágono **existe** → processa só as datas **posteriores** à última, até a
     **última composição VIIRS disponível** para o(s) tile(s) daquele balde
     (consultada no CMR, sem baixar nada);
   - hexágono já atualizado → **pulado**.

   O estado é derivado **do próprio parquet** (`state.load_last_date_by_hexagon`).

4. **Máscaras MapBiomas anuais**: `mapbiomas.prefetch` prepara as máscaras
   (coverage + safrinha) dos anos tocados no run — só os necessários (num run
   incremental, normalmente só o ano corrente). Regra de ano: a máscara é anual
   e, se o VIIRS estiver num ano **posterior** ao último MapBiomas lançado,
   mantém-se o último GeoTIFF (generalizado para *clamp* ao intervalo
   disponível). Fontes das URLs (todos os anos, incl. 2012–2019):
   - **coverage**: URL pública estável da Collection 10 por ano (1985..último;
     último detectado por *listing* do GCS numa única chamada);
   - **safrinha** (`second_crop`): **registro de UUIDs** 2012–2024 embutido
     (colhido da API de export do MapBiomas — UUIDs estáveis), com a API só como
     *fallback* para coleções futuras.

   **Download (padrão) vs. leitura remota**: os GeoTIFFs são COGs, mas a máscara
   é lida **uma vez por balde** e há **poucos arquivos distintos** (um por ano)
   — como `nº de leituras >> nº de arquivos`, **baixar cada COG uma vez e ler do
   disco local é muito mais rápido** que reler remotamente por balde (uma
   leitura remota de um tile inteiro leva ~dezenas de segundos; × ~60 baldes × N
   anos inviabiliza o run completo). Por isso o padrão é **download** (~0,8 GB
   por coverage). Para tocar poucos baldes/anos sem gastar disco,
   `VIIRS_MAPBIOMAS_REMOTE=1` força a leitura remota via `/vsicurl` (aproveita
   os overviews dos COGs), sem download.
   Disco: run completo (2012–2024) ≈ 26 arquivos (~12–13 GB efêmeros); run
   incremental ≈ 1–2 arquivos (~1,6 GB).

5. **Memória**: o cubo do balde é construído **ano a ano** (um ano por vez),
   com buffer float32 pré-alocado (sem `xr.concat` de lista) — teto de RAM ~1
   ano de cubo. Corrige o OOM da versão batch em janelas longas.

6. **Saída incremental durável**: cada balde vira uma **parte**
   (`_parts/balde_<label>.parquet`) enviada ao blob assim que fica pronta. Se o
   pod morrer, o run seguinte **resume** (pula baldes cuja parte já está no
   blob). No fim, canônico + partes são unidos em **streaming** (pyarrow, sem
   carregar o Brasil todo em RAM) no novo `evi_brazil.parquet`, enviado ao blob;
   as partes são limpas.

## Fluxo (entrypoint `files/run.py`)

```
1. pip install -r requirements.txt
2. baixa evi_brazil.parquet (estado) + partes de resume do blob de saída
3. monta o universo de hexágonos (boundary do Brasil no blob de entrada)
4. pré-baixa as máscaras MapBiomas dos anos do run
5. para cada balde: processa incrementalmente -> parte -> upload
6. merge canônico + partes -> evi_brazil.parquet -> upload -> limpa partes
```

## Deploy (kbatch)

`config.yaml` segue o padrão da `../modis-evi-hexagono`. **Segredos não ficam no
YAML** — passe no submit:

```bash
kbatch job submit -f config.yaml \
  -e AZURE_STORAGE_CONNECTION_STRING="$AZURE_STORAGE_CONNECTION_STRING" \
  -e EARTHDATA_USERNAME="$EARTHDATA_USERNAME" \
  -e EARTHDATA_PASSWORD="$EARTHDATA_PASSWORD"
```

Pré-requisitos no blob de **entrada** (`planetary-routines-input`):
- `brazil.geojson` (contorno do Brasil) — ou defina `VIIRS_HEX_LIST_BLOB` com um
  CSV/parquet de `id_hexagono`, ou `VIIRS_BBOX="minx miny maxx maxy"` (teste).

Saída no blob de **saída** (`planetary-routines-output`), prefixo
`VIIRS_EVI_HEXAGONO/`:
- `evi_brazil.parquet` (canônico) e, durante o run, `_parts/balde_*.parquet`.

> **Importante**: o `evi_brazil.parquet` é sempre baixado do blob no início —
> é ele que permite processar só o que falta. Não apague-o entre execuções.

## Variáveis de ambiente

| Variável | Default | Descrição |
|---|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | — | **obrigatória** (I/O no blob) |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | — | login NASA (ou `~/.netrc`) |
| `FENOLOGIA_ACCESS` | `auto` | `s3` (força S3, recomendado in-region), `https`, `auto` |
| `FENOLOGIA_DL_WORKERS` | `4` | concorrência de download VIIRS (↓ reduz RAM) |
| `FENOLOGIA_PREFETCH_DATES` | `2` | janela de prefetch VIIRS (↓ reduz RAM/disco) |
| `VIIRS_MAPBIOMAS_REMOTE` | — | `1` = lê MapBiomas remoto (/vsicurl), sem baixar (mais lento; economiza disco) |
| `VIIRS_START_DATE` | `2012-01-17` | início da série |
| `VIIRS_END_DATE` | hoje | fim da série (fixe p/ reprocessos determinísticos) |
| `VIIRS_BOUNDARY_BLOB` | `brazil.geojson` | contorno no container de input |
| `VIIRS_HEX_LIST_BLOB` | — | alternativa: lista de `id_hexagono` |
| `VIIRS_BBOX` | — | alternativa de teste: `"minx miny maxx maxy"` |
| `VIIRS_LIMIT` | — | processa só N hexágonos (teste) |

## Esquema de saída (`evi_brazil.parquet`, formato *wide*)

Uma linha por `(id_hexagono, data)`, `data` = data de composição nativa do
VIIRS. Colunas `evi_medio_<classe>`, `evi_min_<classe>`, `evi_p25_<classe>`,
`evi_p75_<classe>` para cada classe da macroclasse Agricultura + safrinha
(schema fixo — ver `viirs_evi/config.py`). Idêntico ao da `fenologia`.

## Estrutura

```
viirs-evi-hexagono/
  config.yaml                 # kbatch
  files/
    run.py                    # entrypoint (incremental + blob)
    requirements.txt
    viirs_evi/
      config.py               # parâmetros (env-configurável)
      grid.py  tiles.py  rasterutils.py   # (copiados da fenologia)
      viirs.py                # acesso S3 + cubo por janela [start,end]
      mapbiomas.py            # coverage coll-10 + safrinha (clamp) + pré-download
      extract.py              # estatísticas por classe (copiado)
      state.py                # última data por hexágono (do parquet)
      pipeline.py             # orquestração incremental (ano a ano)
      blob.py                 # Azure Blob I/O
      merge.py                # merge streaming canônico + partes
```

## Custo / escala

O primeiro run com `VIIRS_START_DATE=2012` é pesado (~13 anos × ~46
composições/tile). Runs seguintes são **incrementais**: só as composições novas
desde o último parquet — baratos. Um único pod processa os baldes em série; para
acelerar, dá para particionar por conjunto de hexágonos (vários jobs kbatch com
`VIIRS_HEX_LIST_BLOB` distintos), já que cada balde é independente e o parquet é
mesclado por partes disjuntas.
```
