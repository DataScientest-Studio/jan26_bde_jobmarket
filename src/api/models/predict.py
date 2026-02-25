"""
Modèles de requête et réponse pour les opérations de prédiction ROME.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Requête de prédiction de code ROME pour une offre d'emploi"""
    intitule: Optional[str] = Field(None, description="Titre du poste", example="Développeur Python Senior")
    description: Optional[str] = Field(None, description="Description détaillée du poste", example="Développement d'applications web avec Python, FastAPI, PostgreSQL")
    competences: Optional[List[str]] = Field(None, description="Liste des compétences techniques", example=["Python", "FastAPI", "SQL", "Docker"])
    
    class Config:
        schema_extra = {
            "example": {
                "intitule": "Data Scientist Senior",
                "description": "Analyse de données, machine learning, déploiement de modèles en production",
                "competences": ["Python", "Scikit-learn", "TensorFlow", "SQL"]
            }
        }


class PredictResponse(BaseModel):
    """Résultat de la prédiction avec les codes ROME les plus probables"""
    model_name: Optional[str] = Field(None, description="Nom du modèle utilisé")
    model_version: Optional[str] = Field(None, description="Version du modèle")
    rome_pred: str = Field(..., description="Code ROME prédit (le plus probable)", example="M1805")
    rome_label: Optional[str] = Field(None, description="Libellé du code ROME prédit", example="Études et développement informatique")
    top_k: List[dict] = Field(..., description="Top K prédictions avec scores", example=[
        {"rome_code": "M1805", "score": 0.89, "label": "Études et développement informatique"},
        {"rome_code": "M1806", "score": 0.76, "label": "Conseil et maîtrise d'ouvrage en systèmes d'information"}
    ])
