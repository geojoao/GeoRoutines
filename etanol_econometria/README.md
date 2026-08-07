# O Brasil é tomador ou formulador do preço mundial do etanol?

Estudo econométrico paralelo ao do açúcar (`../acucar_econometria`), agora
cruzando o **preço mundial de referência do etanol** com a **quantidade de
etanol produzida no Brasil**, com dados anuais de **1990 a 2022**.

> **Resposta curta:** o Brasil é **tomador do preço mundial do etanol** — e, em
> alguns aspectos, de forma **mais nítida** do que no açúcar. O preço "global"
> do etanol é, na prática, o **preço americano** (correlação de **0,998** entre
> a referência mundial e o preço dos EUA); o que **realmente move** esse preço é
> o **petróleo** (~46% da variância), não o Brasil (~5%); e, no período pós-2005,
> é o **preço mundial que Granger-causa a produção brasileira** (o Brasil
> reage ao preço), não o contrário. Sobra apenas um traço de "país grande": uma
> **elasticidade preço–produção negativa e significativa (−0,44)** no curto
> prazo, coerente com os ~30% de participação do Brasil — mas que **não** se
> traduz em causalidade nem em fração relevante da variância do preço.
>
> **O arco histórico é o achado central e é o _oposto_ do açúcar:** o Brasil saiu
> de **quase-monopólio** no etanol (era Proálcool: ~99% do mundo em 1980, ~80%
> em 1990) para **~29% (2022)**, ultrapassado pelos EUA (~50%). Ou seja, no
> etanol o Brasil caminhou **de formulador potencial para tomador**; no açúcar,
> ele ganhou participação, mas seguiu tomador.

---

## 1. Por que o etanol é um caso diferente

O açúcar tem **um** preço mundial líquido (o indicador ISA do World Bank). O
etanol **não**: é um mercado **regionalmente segmentado** — milho nos EUA, cana
no Brasil, beterraba/trigo na UE — com comércio limitado por tarifas e logística.
O preço "mundial" de referência é dominado pelo maior produtor. E os dados
mostram isso de forma direta: a referência mundial e o preço americano têm
correlação de **0,998** (figura implícita nos dados). Isto é, o **preço global do
etanol é essencialmente formado no _corn belt_ americano** — antes de qualquer
teste, já é uma evidência forte de que o Brasil não é o formulador.

A pergunta segue idêntica à do açúcar:

| | Causalidade esperada | Sinal | Assinatura |
|---|---|---|---|
| **Tomador** | preço → produção | — | choques de oferta BR não movem o preço |
| **Formulador / país grande** | produção → preço | **negativo** | mais etanol BR ⇒ preço mundial menor |

## 2. Dados

| Série | Fonte | Unidade | Cobertura |
|---|---|---|---|
| **Preço mundial de referência do etanol** | **OECD-FAO** Agricultural Outlook (Aglink-Cosimo), via DBnomics | US$/hectolitro | 1990–2022 (histórico; projeções 2023+ descartadas) |
| Produção BR de etanol | **EIA** International Energy Data | mil barris/dia | 1980–2024 |
| Produção mundial e dos EUA de etanol | EIA | mil barris/dia | 1980–2024 |
| Preço mundial do açúcar (elo da cana) | World Bank Pink Sheet | US$/kg | — |
| Petróleo Brent (elo da gasolina) | World Bank Pink Sheet | US$/bbl | — |
| CPI-U EUA (deflator) | BLS | índice | — |

Preço deflacionado pelo CPI-U para **US$/hl de 2022**. Variáveis em **log**.
Amostra efetiva de estimação: **1990–2022 (n = 33)**.

```bash
pip install -r requirements.txt
python src/build_dataset.py     # gera data/dataset_anual.csv
python src/run_analysis.py      # gera outputs/resultados.json e figuras/
```

## 3. Métodos

Mesma bateria do estudo do açúcar — raiz unitária (ADF/KPSS), cointegração
(Engle-Granger e Johansen), causalidade de Granger em **VAR de diferenças** e
**Toda-Yamamoto**, regressão de **"país grande"** (elasticidade preço–produção,
erros **HAC**) e **IRF/FEVD** — mais um bloco específico do etanol:

- **Nexo cana–gasolina:** a produção brasileira de etanol responde ao **preço do
  açúcar** (custo de oportunidade da cana) e ao **petróleo/gasolina** (substituto
  no bico da bomba flex)?
- **FEVD de 3 variáveis** (produção BR, petróleo, preço do etanol): quem explica
  o preço mundial do etanol?

Dado o tamanho da amostra (n = 33), a seleção de defasagens é limitada a 3.

## 4. Resultados

### 4.1 Ordem de integração
Preço real e produção são ambos **I(1)** (ADF não rejeita em nível — p = 0,30 e
0,25; as primeiras diferenças são estacionárias). Caso mais limpo que o do
açúcar. Correlação de nível entre log-preço e log-produção ≈ **0,00** (contra
−0,43 no açúcar).

### 4.2 Cointegração
Sem cointegração robusta: Engle-Granger não rejeita em nenhum sentido
(p = 0,45 / 0,51) e Johansen **não** rejeita _r_ = 0 (traço 16,4 < 18,4). Não há
elo de equilíbrio de longo prazo preço–produção.

### 4.3 Causalidade de Granger (o teste central)

| Teste | Q → P (produção causa preço?) | P → Q (preço causa produção?) |
|---|---|---|
| VAR em diferenças (lag 1) | F = 0,25 · **p = 0,62** | F = 0,16 · **p = 0,69** |
| Toda-Yamamoto (nível) | F = 0,36 · **p = 0,55** | F = 0,01 · **p = 0,94** |
| **Subamostra pós-2005** | F = 2,22 · p = 0,13 | **F = 3,45 · p = 0,046** |

No período todo, nenhum sentido é significativo. Mas na **era pós-2005** (Brasil
já como nº 2, pós-mandatos americanos do RFS), o **preço mundial Granger-causa a
produção brasileira** (p = 0,046) — exatamente a assinatura do **tomador que
reage ao preço**. O sentido inverso (Brasil → preço) segue não significativo.

### 4.4 O quanto o Brasil "move" o preço

- **FEVD (2 variáveis):** a produção brasileira explica **~5,4%** da variância de
  previsão do preço do etanol.
- **FEVD (3 variáveis — produção BR, petróleo, preço):** **petróleo ~46%**,
  próprio ~48%, **Brasil ~5,5%**. O preço mundial do etanol é governado pelo
  **petróleo**, não pelo Brasil.
- **Elasticidade preço–produção (país grande, HAC):**

  | Especificação | β (Δlq) | p-valor |
  |---|---|---|
  | Contemporânea | **−0,44** | **0,044** |
  | Contemp. + defasagem (soma) | −0,52 | contemp. p = 0,074 |

  Aqui há o **único** sinal de poder de mercado: a elasticidade contemporânea é
  **negativa e significativa a 5%** — coerente com um país que responde por ~30%
  da oferta operar sobre uma demanda residual inclinada. Mas o efeito é **frágil**
  (R² ≈ 0,06), **não** aparece como causalidade de Granger e **não** aparece na
  FEVD. É co-movimento de curto prazo, não formação de preço.

### 4.5 Nexo cana–gasolina
A produção brasileira de etanol **não** responde (a 5%) ao preço defasado do
açúcar (p = 0,60), do petróleo (p = 0,90) nem do próprio etanol (p = 0,94);
R² ≈ 0,01. No agregado anual, a oferta brasileira de etanol é governada por
**capacidade instalada, disponibilidade de cana e política doméstica de mistura**
— e não por arbitragem de preços de curto prazo. Isso reforça o quadro: a
produção **não é o instrumento** com que o Brasil formularia um preço.

## 5. Interpretação — tomador ou formulador?

Convergência para **tomador de preço**, com evidência mais direta que no açúcar:

1. **O preço "global" é o preço dos EUA** (correlação 0,998) — o formulador, se
   existe, está no _corn belt_, não no Centro-Sul.
2. **O petróleo domina o preço** (~46% da variância); o Brasil, ~5%.
3. **Pós-2005 o preço Granger-causa a produção brasileira** — o Brasil reage ao
   preço, comportamento de tomador.
4. Sem causalidade produção→preço e sem cointegração.

A única nuance é a **elasticidade contemporânea negativa e significativa (−0,44)**:
os ~30% de participação do Brasil não são desprezíveis e geram um co-movimento
de curto prazo com o sinal que o poder de mercado prevê. Mas isso **não** se
sustenta como causalidade nem como fração da variância — é influência marginal,
não formulação.

### Veredito

> **O Brasil é tomador do preço mundial do etanol.** O preço de referência é
> formado nos EUA e movido pelo petróleo; a produção brasileira reage ao preço
> (pós-2005) em vez de determiná-lo, não o Granger-causa e explica ~5% de sua
> variância. O resíduo de poder de mercado (elasticidade −0,44 no curtíssimo
> prazo) reflete o tamanho do Brasil, mas não o qualifica como formulador.

### Comparação com o açúcar

| | **Açúcar** | **Etanol** |
|---|---|---|
| Existe preço mundial único? | Sim (ISA) | Não — segmentado; referência ≈ preço EUA |
| Participação do Brasil | **subindo** (~7%→22%) | **caindo** (~80%→29%; EUA nº 1) |
| Granger produção→preço | não sig. | não sig. |
| Granger preço→produção | não sig. | **sig. pós-2005** |
| Elasticidade preço–produção | −0,11 (n.s.) | **−0,44 (sig. 5%)** |
| Variância do preço explicada pelo BR | ~2% | ~5% (petróleo ~46%) |
| **Veredito** | Tomador (influência latente crescente) | **Tomador (mais nítido; trajetória de formulador→tomador)** |

Nos dois mercados o Brasil é **tomador**. A diferença é a **direção do tempo**:
no açúcar sua participação e sua influência latente **crescem**; no etanol ele
**perdeu** a posição dominante que teve na era Proálcool e hoje segue um preço
formado fora do país.

## 6. Limitações
- **Amostra curta** (33 anos) e anual → baixo poder; a elasticidade significativa
  de −0,44 deve ser lida com cautela.
- O preço de referência OECD-FAO é um **preço-modelo** (Aglink-Cosimo), não uma
  cotação de bolsa; um teste mais forte usaria **futuros mensais** (CBOT ethanol,
  B3 hidratado) e **volumes de exportação**, com **instrumentos** de oferta
  (clima no Centro-Sul, mandatos RenovaBio/RFS, câmbio, paridade do etanol).
- A produção (EIA) e o preço (OECD-FAO) vêm de fontes distintas; um exercício de
  robustez usaria a produção do próprio Aglink-Cosimo.

## 7. Arquivos
```
etanol_econometria/
├── README.md
├── requirements.txt
├── data/
│   ├── ethanol_price_oecdfao.csv        # preço mundial de ref. + EUA (OECD-FAO)
│   ├── ethanol_production_eia_tbpd.csv  # produção BR/EUA/mundo (EIA)
│   ├── sugar_price_world_annual.csv     # elo cana (World Bank)
│   ├── brent_oil_annual.csv             # elo gasolina (World Bank)
│   ├── us_cpi_annual.csv                # deflator (BLS)
│   └── dataset_anual.csv                # painel final
├── src/
│   ├── build_dataset.py
│   └── run_analysis.py
└── outputs/
    ├── resultados.json
    └── figuras/                         # séries+participação, IRF, elasticidade
```

**Fontes:** OECD-FAO Agricultural Outlook 2023-2032 / Aglink-Cosimo (via
DBnomics); U.S. EIA International Energy Data; World Bank "Pink Sheet"; U.S. BLS.
