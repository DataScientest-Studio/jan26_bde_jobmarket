import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field

# IMPORTANT: import depuis ton module de prédiction
# ajuste le chemin selon ton projet (ex: from src.models.predict_model import ...)
from src.models.predict_model import build_text_payload, load_artifacts, predict_top_k, get_rome_model

MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI(title="ROME Classifier API", version="1.0.0")

# Cache global (chargé une seule fois)
ARTIFACTS: Dict[str, Any] = {}


class PredictRequest(BaseModel):
    intitule: Optional[str] = None
    description: Optional[str] = None
    competences: Optional[List[str]] = Field(default=None, description="Liste de compétences (optionnel)")


class PredictResponse(BaseModel):
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    rome_pred: str
    rome_label: Optional[str] = None
    top_k: List[dict]

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": ARTIFACTS.get("version"),
        "storage_backend": os.getenv("STORAGE_BACKEND", "local"),
    }

global rome_model
rome_model = get_rome_model()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = build_text_payload(
        intitule=req.intitule,
        description=req.description,
        competences=req.competences,
    )

    pred = predict_top_k(ARTIFACTS, text, top_k=TOP_K, rome_index=rome_model)
    return pred
    return {
        "model_name": MODEL_NAME,
        "model_version": ARTIFACTS["version"],
        "rome_pred": pred["rome_pred"],
        "top_k": pred["top_k"],
    }
