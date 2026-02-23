import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

# IMPORTANT: import depuis ton module de prédiction
# ajuste le chemin selon ton projet (ex: from src.models.predict_model import ...)
from src.models.predict_model import build_text_payload, load_artifacts, predict_top_k, get_rome_model
from src.ingest.bronze.france_travail_rome_metiers import ingest_rome_metiers

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI(
    title="ROME Classifier & Data Ingestion API",
    version="1.0.0",
    description="""
API microservice pour la classification automatique des offres d'emploi selon le référentiel ROME 
et l'ingestion des données de référence depuis France Travail.

## Fonctionnalités

- **Prédiction ROME** : Classifie une offre d'emploi et retourne les codes ROME correspondants
- **Ingestion des codes ROME** : Importe la nomenclature complète des métiers ROME
- **Monitoring** : Status et health checks de l'API

## Utilisation

Consultez les endpoints ci-dessous pour tester l'API directement depuis cette interface.
    """,
    contact={
        "name": "Job Market API Support"
    }
)

# Cache global (chargé une seule fois)
ARTIFACTS: Dict[str, Any] = {}


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


class IngestResponse(BaseModel):
    """Résultat d'une opération d'ingestion de données"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    key: Optional[str] = Field(None, description="Clé de stockage des données", example="bronze/rome/rome_metiers.jsonl")
    records_count: Optional[int] = Field(None, description="Nombre total de codes ROME", example=532)
    records_written: Optional[int] = Field(None, description="Nombre d'enregistrements écrits", example=532)
    error: Optional[str] = Field(None, description="Message d'erreur si échec")

@app.on_event("startup")
def _startup_load_model():
    """
    Charge les artefacts une seule fois au démarrage.
    Evite de recharger MinIO/joblib à chaque requête.
    """
    global ARTIFACTS
    ARTIFACTS = load_artifacts()
    # Optionnel: log
    print(f"✅ API loaded model: {MODEL_NAME} / {ARTIFACTS['version']}")


@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Health check de l'API",
    description="Vérifie que l'API est opérationnelle et retourne les informations sur le modèle chargé"
)
def health():
    """Health check de l'API avec informations sur le modèle et le backend de storage"""
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": ARTIFACTS.get("version"),
        "storage_backend": os.getenv("STORAGE_BACKEND", "local"),
    }

global rome_model
rome_model = get_rome_model()


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Classification ROME"],
    summary="Prédire le code ROME d'une offre d'emploi",
    description="""Classifie automatiquement une offre d'emploi selon le référentiel ROME.

Fournissez au minimum un intitulé ou une description de poste.
Le modèle retourne le code ROME le plus probable ainsi qu'un top-K des codes les plus pertinents.
"""
)
def predict(req: PredictRequest):
    """Classification d'une offre d'emploi selon le référentiel ROME.

**Paramètres:**

- **intitule**: Titre du poste (optionnel mais recommandé)
- **description**: Description détaillée (optionnel mais recommandé)
- **competences**: Liste de compétences techniques (optionnel)

**Exemple:**

```json
{
  "intitule": "Data Scientist",
  "description": "Analyse de données et machine learning",
  "competences": ["Python", "Scikit-learn", "SQL"]
}
```
"""
    text = build_text_payload(
        intitule=req.intitule,
        description=req.description,
        competences=req.competences,
    )

    pred = predict_top_k(ARTIFACTS, text, top_k=TOP_K, rome_index=rome_model)
    return pred


# =====================================
# Endpoints d'ingestion
# =====================================

@app.post(
    "/ingest/rome-metiers",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingérer les codes ROME métiers",
    description="""Déclenche l'ingestion de la nomenclature complète des codes ROME métiers depuis l'API France Travail.

Les données sont stockées dans `bronze/rome/rome_metiers.jsonl` (environ 532 codes ROME).

**Modes d'exécution:**

- **Synchrone** (background=false): Attend la fin de l'ingestion et retourne le résultat
- **Asynchrone** (background=true): Lance l'ingestion en arrière-plan et retourne immédiatement

**Utilisation:**

```bash
# Synchrone
curl -X POST "http://localhost:8000/ingest/rome-metiers"

# Asynchrone
curl -X POST "http://localhost:8000/ingest/rome-metiers?background=true"
```
"""
)
async def ingest_rome_metiers_endpoint(
    background: bool = Query(False, description="Lancer en arrière-plan"),
    background_tasks: BackgroundTasks = None
):
    """Ingestion des codes ROME métiers depuis l'API France Travail.

Récupère tous les codes ROME métiers avec leurs libellés et les stocke
dans le système de storage configuré (local ou S3/MinIO).
"""
    logger.info(f"Requête d'ingestion ROME métiers reçue (background={background})")
    
    if background and background_tasks:
        # Lancer en arrière-plan
        logger.info("Lancement de l'ingestion en arrière-plan")
        background_tasks.add_task(ingest_rome_metiers)
        return IngestResponse(
            success=True,
            message="Ingestion des codes ROME métiers lancée en arrière-plan"
        )
    else:
        # Exécution synchrone
        try:
            logger.info("Début de l'ingestion synchrone")
            result = ingest_rome_metiers()
            
            if result["success"]:
                logger.info(f"Ingestion réussie: {result['records_count']} codes ROME")
            else:
                logger.error(f"Échec de l'ingestion: {result.get('error')}")
                
            return IngestResponse(**result)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ingestion: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Erreur lors de l'ingestion: {str(e)}"
            )


@app.get(
    "/ingest/status",
    tags=["Ingestion"],
    summary="Statut des opérations d'ingestion",
    description="Liste toutes les opérations d'ingestion disponibles et leur statut"
)
def get_ingest_status():
    """Retourne la liste des opérations d'ingestion disponibles.

Utile pour découvrir les endpoints d'ingestion et vérifier la disponibilité du service.
"""
    return {
        "status": "ok",
        "available_operations": [
            {
                "endpoint": "POST /ingest/rome-metiers",
                "description": "Ingestion des codes ROME métiers depuis France Travail",
                "params": ["background (bool, optionnel)"]
            }
        ]
    }
