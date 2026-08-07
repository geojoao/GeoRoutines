# O Brasil é tomador ou formulador do preço mundial do açúcar?

Estudo econométrico da relação entre a **série do preço mundial do açúcar** e a
**quantidade de açúcar produzida no Brasil**, com dados anuais de **1961 a 2023**.

> **Resposta curta:** com os dados anuais, o Brasil se comporta
> **muito mais como um _tomador_ de preço (price taker) do que como um
> formulador (price maker)** — apesar de ser o maior produtor mundial. A
> produção brasileira **não causa (Granger) nem move de forma
> estatisticamente detectável** o preço mundial, e explica apenas **~2%** da
> variância do preço. Há, porém, um traço consistente de "país grande": o
> **sinal** da relação preço–produção é sempre **negativo** (o sinal que o poder
> de mercado prevê), mas **impreciso demais** para sustentar a tese de
> formulador. A leitura honesta é: **tomador de preço com influência latente
> crescente**, não formulador.

---

## 1. A pergunta e a estratégia de identificação

Duas hipóteses geram previsões observacionalmente distintas:

| | Mecanismo | Causalidade esperada | Sinal | Assinatura nos dados |
|---|---|---|---|---|
| **Tomador de preço** | Brasil é "pequeno"; o preço mundial é exógeno | preço → produção (o Brasil reage ao preço) | — | choques de oferta brasileiros **não** movem o preço |
| **Formulador / país grande** | O Brasil desloca a demanda residual do resto do mundo | produção → preço | **negativo** | mais açúcar brasileiro ⇒ preço mundial **menor** |

A intuição de "país grande": se o Brasil é grande o suficiente, ele opera sobre
a curva de **demanda residual** do mundo (demanda mundial menos oferta dos
demais). Nesse caso, um aumento da oferta brasileira empurra o preço mundial
**para baixo** (elasticidade preço–produção negativa e significativa). Se o
Brasil for um tomador, o preço é exógeno a ele e é a **produção** que responde
ao preço.

Isso **não** é um detalhe acadêmico: o Brasil é o **maior produtor e maior
exportador** mundial de açúcar. Sua participação na produção mundial saltou de
**~7% (1961) para ~22% (2023)** (ver figura 1). A priori isso sugeriria poder de
mercado — a econometria testa se esse poder aparece de fato nos preços.

## 2. Dados

| Série | Fonte | Unidade | Cobertura |
|---|---|---|---|
| Preço mundial do açúcar ("Sugar, world") | World Bank **Pink Sheet** (CMO) | US$/kg nominal, mensal → média anual | 1960–2024 |
| Produção BR de **açúcar** (bruto centrifugado) | **FAOSTAT** (QCL) | toneladas | 1961–2023 |
| Produção BR de **cana** (robustez) | FAOSTAT (QCL) | toneladas | 1961–2024 |
| Produção **mundial** de açúcar (participação) | FAOSTAT (QCL) | toneladas | 1961–2023 |
| CPI-U EUA (deflator → preço real) | BLS (média anual, 1982-84=100) | índice | 1960–2024 |

O preço é deflacionado pelo CPI-U para **US$/kg de 2023**. A variável de
quantidade principal é o **açúcar bruto centrifugado** (o produto de fato), com a
**cana** como checagem de robustez. Todas as séries entram em **log**.

Reprodução:

```bash
pip install -r requirements.txt
python src/build_dataset.py     # gera data/dataset_anual.csv
python src/run_analysis.py      # gera outputs/resultados.json e figuras/
```

Os arquivos brutos usados estão em `data/` (Pink Sheet, FAOSTAT e CPI já
extraídos e limpos), de modo que a análise roda offline.

## 3. Métodos

1. **Raiz unitária** — ADF e KPSS em nível e em primeira diferença, para fixar a
   ordem de integração.
2. **Cointegração** — Engle-Granger (nos dois sentidos) e Johansen (traço).
3. **Causalidade de Granger** — VAR bivariado em diferenças (defasagens por AIC),
   testando os **dois sentidos**; e **Toda-Yamamoto (1995)** em nível
   (VAR com _p_+_d_max_ defasagens, Wald), que é **robusto à ambiguidade de
   integração/cointegração**.
4. **Regressão de "país grande"** — elasticidade do preço mundial à produção
   brasileira, Δlp = α + β·Δlq (+ defasagens), com erros **HAC (Newey-West)**.
5. **IRF e FEVD** — resposta do preço a um choque de produção e fração da
   variância do preço explicada pela produção (ordenação Δq → Δp, tratando a
   produção como predeterminada no ano — a cana é decidida safras antes).
6. **Robustez** — cana no lugar do açúcar; preço nominal; subamostra pós-1990.

## 4. Resultados

### 4.1 Ordem de integração

| Série | ADF (p) | KPSS (p) | Leitura |
|---|---|---|---|
| log preço real (nível) | **0,022** | 0,10 | ~estacionária em torno de tendência |
| log produção BR (nível) | 0,273 | 0,10 | tendência forte → tratada como I(1) |
| Δ log preço real | 0,000 | 0,10 | I(0) |
| Δ log produção BR | 0,018 | 0,10 | I(0) |

O preço real é praticamente **estacionário em tendência**, enquanto a produção é
**I(1)**. Já esse contraste é informativo: séries de ordens diferentes
dificilmente compartilham um equilíbrio de longo prazo estável — o que
enfraquece, de saída, a ideia de um vínculo de "formação de preço" comandado
pelo Brasil.

### 4.2 Cointegração

- Engle-Granger: rejeita a não-cointegração **só** quando o preço é a variável
  dependente (p = 0,039); não rejeita no sentido inverso (p = 0,458) — resultado
  **frágil e sensível à normalização**.
- Johansen (traço): rejeita _r_=0 **e** _r_≤1 a 95% → aponta para um sistema
  **estacionário** (posto cheio), coerente com o preço ser I(0) e **não** com um
  único vetor de cointegração.

**Conclusão:** não há evidência robusta de tendência estocástica comum. Não há
um "elo de longo prazo" preço–produção que o Brasil ancore.

### 4.3 Causalidade de Granger (o teste central)

| Teste | Q → P (produção causa preço?) | P → Q (preço causa produção?) |
|---|---|---|
| VAR em diferenças (lag AIC=1) | F=1,05 · **p=0,31** | F=0,10 · **p=0,75** |
| Toda-Yamamoto (nível, p=1, d=1) | F=0,59 · **p=0,45** | F=0,15 · **p=0,70** |

**Nenhum** sentido causal é significativo. Nem a produção brasileira antecipa o
preço mundial (o que o formulador exigiria), nem o preço antecipa a produção (o
que o tomador "reativo" exigiria). No agregado anual, as duas séries são
dinamicamente quase **desacopladas**.

### 4.4 O quanto o Brasil "move" o preço

- **FEVD:** a produção brasileira explica apenas **~2,0%** da variância de
  previsão do preço mundial no horizonte de 10 anos. (Figuras 2 e 3.)
- **Elasticidade preço–produção** (regressão de país grande, erros HAC):

  | Especificação | β (Δlq) | p-valor |
  |---|---|---|
  | Contemporânea | −0,11 | 0,82 |
  | Contemp. + 1 defasagem (soma) | −0,59 | n.s. |
  | Só defasada (Δlq₋₁) | −0,46 | 0,32 |

  O **sinal é sempre negativo** — exatamente o que o poder de mercado prevê —,
  mas **nunca é estatisticamente distinguível de zero**. Ou seja: há a *direção*
  de um país grande, sem a *precisão* para afirmá-lo.

### 4.5 Robustez

| Variante | Q → P (p) | P → Q (p) | Comentário |
|---|---|---|---|
| Cana no lugar do açúcar | 0,73 | 0,18 | idem: sem causalidade |
| **Preço nominal** | **0,049** | 0,21 | único "significativo"; provável **espúrio** (tendências comuns / inflação em nível) |
| Pós-1990 | 0,91 | 0,28 | idem, apesar do share já alto |

O único resultado "significativo" aparece com o **preço nominal** (lag 5) — e
some quando se deflaciona, sinal clássico de correlação espúria entre séries que
sobem juntas por inflação. Não sustenta a tese de formulador.

## 5. Interpretação — tomador ou formulador?

O peso das evidências aponta para **tomador de preço** no plano macro anual:

1. **Sem causalidade de Granger robusta** em nenhum sentido (VAR e
   Toda-Yamamoto) → a produção brasileira não prevê o preço mundial.
2. **~2% da variância** do preço explicada pela produção brasileira → impacto
   econômico trivial.
3. **Elasticidade insignificante** (embora com o sinal negativo "certo").

Mas três nuances impedem o rótulo de tomador *puro* e apontam para poder de
mercado **latente**:

- O **sinal** da relação preço–produção é sistematicamente **negativo** (inclui
  a correlação de nível de −0,43), a assinatura de uma demanda residual
  inclinada — só que fraca demais para ser conclusiva.
- A **participação triplicou para ~22%**: estruturalmente o Brasil *é* um país
  grande. A insignificância pode refletir **baixo poder do teste** (~60
  observações anuais) mais do que ausência real de influência.
- Boa parte do canal de mercado do Brasil opera via **exportações e desvio
  cana→etanol** em frequência **intra-anual** (safra, paridade do etanol), que o
  dado anual **suaviza**. A expansão brasileira, aliás, ocorreu **sob preços
  reais baixos e decrescentes** (figura 1) — coerente com o Brasil empurrando o
  preço ao longo da demanda residual, ainda que o preço seja dominado por outros
  fatores globais (clima, Índia, UE, petróleo/etanol, políticas comerciais).

### Veredito

> **No horizonte anual, o Brasil age predominantemente como _tomador_ do preço
> mundial do açúcar.** Sua produção não formula nem antecipa o preço de forma
> estatisticamente detectável e responde por apenas ~2% da sua variância. O
> poder de mercado que seu tamanho (~⅕ do mundo) sugeriria aparece apenas como
> um **sinal negativo consistente, porém não significativo** — uma influência
> latente que cresce com o *share*, mas que **não** o qualifica, com estes
> dados, como formulador de preço.

## 6. Limitações e extensões

- **Frequência e amostra:** ~60 observações anuais têm baixo poder. O teste
  natural mais forte usa **preços futuros mensais** (ICE nº 11) e **volumes de
  exportação** brasileiros.
- **Endogeneidade/identificação:** a estimativa limpa do efeito da oferta
  brasileira sobre o preço pede **instrumentos** de oferta — clima (chuva/geada
  no Centro-Sul), **paridade açúcar–etanol** e câmbio — que deslocam a oferta
  sem responder ao preço mundial. É a extensão mais valiosa.
- **Demanda residual estrutural** (Goldberg–Knetter): estimar diretamente a
  elasticidade da demanda residual enfrentada pelo Brasil com esses
  instrumentos daria o teste de poder de mercado mais direto.

## 7. Arquivos

```
acucar_econometria/
├── README.md                     # este relatório
├── requirements.txt
├── data/
│   ├── sugar_price_world_annual.csv / _monthly.csv   # World Bank Pink Sheet
│   ├── brazil_sugar_production_fao.csv               # FAOSTAT (açúcar e cana BR)
│   ├── world_sugar_production_fao.csv                # FAOSTAT (mundo)
│   ├── us_cpi_annual.csv                             # deflator (BLS CPI-U)
│   └── dataset_anual.csv                             # painel final montado
├── src/
│   ├── build_dataset.py          # integra e deflaciona
│   └── run_analysis.py           # toda a econometria + figuras
└── outputs/
    ├── resultados.json           # todas as estatísticas
    └── figuras/                  # 3 figuras (séries, IRF, elasticidade)
```

**Fontes:** World Bank Commodity Price Data — "The Pink Sheet" (atualização de
jan/2025); FAOSTAT — Production/Crops & Livestock (QCL); U.S. BLS — CPI-U.
