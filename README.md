# AT2 — Redes Neurais MLP e Deploy de API em Produção

Classificação de objetos celestes do **Sloan Digital Sky Survey (SDSS17)** com Rede Neural Multicamadas (`MLPClassifier`), empacotada em `Pipeline` do Scikit-Learn e disponibilizada via **FastAPI**.

## Dataset

- **Fonte:** [Stellar Classification Dataset — SDSS17 (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
- **Arquivo:** `data/star_classification.csv`
- **Target:** `class` (multiclasse: `GALAXY`, `QSO`, `STAR`)
- **Features:** `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift` (numéricas) + `faixa_redshift` (categórica derivada de `redshift`)
- **Removidas:** identificadores (`obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID`)

## Estrutura do repositório

```text
am-at2-redes-neurais/
├── data/
│ └── star_classification.csv
├── models/
│ └── pipeline_rede_neural.joblib
├── notebooks/
│ └── analise_treinamento.ipynb
├── src/
│ ├── init.py
│ ├── schemas.py
│ └── app.py
├── pyproject.toml
├── requisicoes.http
└── README.md
```


## Modelo

- **Pré-processamento:** `ColumnTransformer` com `StandardScaler` (numéricas) + `OneHotEncoder(handle_unknown="ignore")` (categórica)
- **Rede:** `MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam", learning_rate_init=0.001, max_iter=500, random_state=77, early_stopping=True, n_iter_no_change=15)`
- **Split:** 80/20 estratificado, `random_state=77`
- **Artefato:** pipeline completo serializado em `models/pipeline_rede_neural.joblib` (contém scaler + encoder + rede — a API recebe dados brutos)

### Métricas no conjunto de teste

- **Acurácia global:** [preencher com o valor do notebook]
- **Matthews Correlation Coefficient (MCC):** [preencher]
- **Relatório por classe:** ver `notebooks/analise_treinamento.ipynb` (célula de avaliação)
- **Matriz de confusão:** ver notebook

## Como reproduzir

### 1. Instalar dependências (UV)

```bash
uv sync
```

### 2. Treinar e exportar o modelo

  Abra `notebooks/analise_treinamento.ipynb` e execute todas as células (Kernel → Restart & Run All). Isso gera `models/pipeline_rede_neural.joblib`
  
### 3. Subir a API

```bash
uv run fastapi dev src/app.py
```
  Swagger UI: http://127.0.0.1:8000/docs

  Redoc: http://127.0.0.1:8000/redoc

## Endpoints
`GET /diagnostico/modelo`

Metadados e status do serviço.

```json
{
  "status": "operacional_v2",
  "framework": "FastAPI-Production",
  "modelo": "MLPClassifier"
}
```

`POST /predict`

Inferência a partir dos dados brutos (validados pelo Pydantic).

Request:
```json
{
  "alpha": 135.7,
  "delta": 32.1,
  "u": 22.5,
  "g": 21.0,
  "r": 20.4,
  "i": 19.9,
  "z": 19.6,
  "redshift": 0.12,
  "faixa_redshift": "medio"
}
```
**Response (200)**:
```json

{
  "categoria_prevista": "GALAXY",
  "score_confianca": 0.9749
}
```
Entrada inválida (ex: texto em campo numérico) retorna 422 com detalhe do Pydantic.

## Testes de integração

O arquivo `requisicoes.http` contém 4 cenários (1 diagnóstico + 2 predições válidas com classes distintas + 1 inválida → 422). Use a extensão REST Client (VS Code) ou Thunder Client para executar. Todas as requisições enviam `User-Agent: Tester-Audit-Agent.`

## Arquitetura da inferência

O pipeline `.joblib` é carregado uma única vez no `lifespan` do FastAPI (startup) e mantido em memória.

As rotas nunca leem disco — apenas chamam `predict`/`predict_proba` sobre o objeto já carregado.

Como o `Pipeline` serializado inclui `StandardScaler` e `OneHotEncoder`, a API recebe dados brutos e aplica as mesmas transformações do treino, sem risco de inconsistência.
