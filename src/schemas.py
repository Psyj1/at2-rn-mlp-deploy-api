# src/schemas.py
from pydantic import BaseModel, Field

# Limites reais observados no dataset (ajuste se quiser refinar depois)
class ObjetoStellarInput(BaseModel):
    """Atributos brutos de um objeto celeste para classificação."""

    alpha: float = Field(
        ..., description="Ascensão reta (graus)", ge=0.0, le=360.0, examples=[135.7]
    )
    delta: float = Field(
        ..., description="Declinação (graus)", ge=-90.0, le=90.0, examples=[32.1]
    )
    u: float = Field(..., description="Magnitude na banda u", ge=-30.0, le=40.0, examples=[22.5])
    g: float = Field(..., description="Magnitude na banda g", ge=-30.0, le=40.0, examples=[21.0])
    r: float = Field(..., description="Magnitude na banda r", ge=-30.0, le=40.0, examples=[20.4])
    i: float = Field(..., description="Magnitude na banda i", ge=-30.0, le=40.0, examples=[19.9])
    z: float = Field(..., description="Magnitude na banda z", ge=-30.0, le=40.0, examples=[19.6])
    redshift: float = Field(
        ..., description="Desvio para o vermelho", ge=-0.1, le=10.0, examples=[0.12]
    )
    faixa_redshift: str = Field(
        ...,
        description="Faixa categórica de redshift (baixo, medio, alto)",
        examples=["medio"],
    )


class PredicaoStellarOutput(BaseModel):
    """Resposta da inferência."""

    categoria_prevista: str = Field(..., description="Classe inferida pelo modelo")
    score_confianca: float = Field(
        ..., ge=0.0, le=1.0, description="Confiança da predição (probabilidade máxima)"
    )