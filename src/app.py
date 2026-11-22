# src/app.py
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.schemas import ObjetoStellarInput, PredicaoStellarOutput

# Estado global: modelo residente na RAM
recursos: dict = {}


def _encontrar_raiz(inicio: Path) -> Path:
    atual = inicio.resolve()
    for pai in [atual, *atual.parents]:
        if (pai / "models" / "pipeline_rede_neural.joblib").exists():
            return pai
    raise FileNotFoundError(
        "Não encontrei models/pipeline_rede_neural.joblib. "
        "Execute o notebook analise_treinamento.ipynb para gerar o artefato."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o pipeline uma única vez no startup (proibido ler disco na rota)."""
    caminho = _encontrar_raiz(Path.cwd()) / "models" / "pipeline_rede_neural.joblib"
    recursos["modelo"] = joblib.load(caminho)
    print(f"[STARTUP] Modelo carregado: {caminho}")
    yield
    recursos.clear()
    print("[SHUTDOWN] Memória liberada.")


app = FastAPI(
    title="API de Classificação Stellar (MLP)",
    description="Inferência com pipeline Scikit-Learn (ColumnTransformer + MLPClassifier)",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/diagnostico/modelo", tags=["Diagnóstico"])
def diagnostico_modelo():
    """Metadados e status do serviço (JSON fixo exigido pela atividade)."""
    return {
        "status": "operacional_v2",
        "framework": "FastAPI-Production",
        "modelo": "MLPClassifier",
    }


@app.post("/predict", response_model=PredicaoStellarOutput, tags=["Predição"])
def prever_objeto(objeto: ObjetoStellarInput):
    """Executa a inferência a partir dos dados brutos validados pelo Pydantic."""
    modelo = recursos.get("modelo")
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não disponível em memória.",
        )

    # Converte o schema validado em DataFrame compatível com o pipeline
    df_entrada = pd.DataFrame([objeto.model_dump()])

    categoria = str(modelo.predict(df_entrada)[0])
    probabilidades = modelo.predict_proba(df_entrada)[0]
    score = float(probabilidades.max())

    return PredicaoStellarOutput(
        categoria_prevista=categoria,
        score_confianca=round(score, 4),
    )