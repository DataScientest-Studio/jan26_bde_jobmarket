import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.config.env import load_project_env

from src.storage.storage import get_storage_from_env

load_project_env()  # safe à rappeler (idempotent)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
MODEL_PATH_PREFIX = os.getenv("MODEL_PATH_PREFIX", "models")  # Can be "models" or "gold/models"
TOP_K = int(os.getenv("TOP_K", "5"))

# Build unified storage backends  
# For models/: use gold layer storage (models are in gold/models/...)
# For rome/: use FT bronze layer storage
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower().strip()

# Get storage for gold layer where models are stored
storage_gold = get_storage_from_env("gold")  # gold/models/...
storage_ft = get_storage_from_env("bronze", "france_travail")  # bronze/france_travail/...

# -----------------------------
# Get ROME model
# -----------------------------

def load_joblib(key: str) -> Any:
    """Load joblib object from gold storage (models layer)"""
    data = storage_gold.read_bytes(key)
    return joblib.load(io.BytesIO(data))

def read_jsonl(key: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dicts from FT storage"""
    return list(storage_ft.read_jsonl(key))

def get_rome_model() -> Dict[str, str]:
    """
    Load ROME codes from storage_ft (bronze/france_travail layer).
    Relative key: "rome/rome_metiers.jsonl"
    """
    try:
        rome_rows = list(storage_ft.read_jsonl("rome/rome_metiers.jsonl"))
        rome_index = {row["code"]: row["libelle"] for row in rome_rows}
        return rome_index
    except Exception as exc:
        raise exc


def safe_get_rome_model() -> Dict[str, str]:
    try:
        rome_index = get_rome_model()
        print(f"[OK] Loaded ROME model: {len(rome_index)} entries")
        return rome_index
    except Exception as exc:
        print(
            "[WARN] ROME model unavailable at startup; continuing with empty index. "
            f"Reason: {exc}"
        )
        return {}


global rome_model
rome_model = safe_get_rome_model()





# -----------------------------
# Text building (same as dataset)
# -----------------------------
_whitespace_re = re.compile(r"\s+")


def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = _whitespace_re.sub(" ", s)
    return s.strip()


def build_text_payload(
    *,
    intitule: Optional[str],
    description: Optional[str],
    competences: Optional[List[str]] = None,
) -> str:
    parts = []
    title = clean_text(intitule)
    desc = clean_text(description)

    if title:
        parts.append(f"[TITRE] {title}")
    if desc:
        parts.append(f"[DESC] {desc}")

    # For scraped jobs, competences may be embedded in description -> this section can be omitted
    if competences:
        # dedup + clean
        seen = set()
        cleaned = []
        for c in competences:
            cc = clean_text(c)
            if not cc:
                continue
            k = cc.lower()
            if k not in seen:
                seen.add(k)
                cleaned.append(cc)
        if cleaned:
            parts.append("[COMP] " + " ".join(cleaned))

    return "\n".join(parts).strip()

# -----------------------------
# Load latest model artifacts
# -----------------------------
def get_latest_version() -> str:
    """Get latest model version from LATEST.json in gold/models/"""
    key = f"models/{MODEL_NAME}/LATEST.json"
    try:
        data = storage_gold.read_bytes(key)
        latest = json.loads(data.decode("utf-8"))
        return latest["latest"]
    except Exception as exc:
        raise FileNotFoundError(f"LATEST.json not found at gold/{key}") from exc


def load_artifacts() -> Dict[str, Any]:
    """Load model artifacts (vectorizer, model, label_encoder) from gold/models/"""
    version = get_latest_version()
    base = f"models/{MODEL_NAME}/versions/{version}"
    
    try:
        vectorizer = load_joblib(f"{base}/vectorizer.joblib")
        model = load_joblib(f"{base}/model.joblib")
        label_encoder = load_joblib(f"{base}/label_encoder.joblib")
        
        return {
            "version": version,
            "base": base,
            "vectorizer": vectorizer,
            "model": model,
            "label_encoder": label_encoder,
        }
    except Exception as exc:
        raise FileNotFoundError(f"Could not load model artifacts from gold/{base}") from exc


# -----------------------------
# Predict
# -----------------------------
def predict_top_k(
    artifacts: Dict[str, Any],
    text: str,
    top_k: int = 5,
    rome_index: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    vectorizer = artifacts["vectorizer"]
    model = artifacts["model"]
    le = artifacts["label_encoder"]

    X = vectorizer.transform([text])

    # LinearSVC: decision scores (not probabilities)
    scores = model.decision_function(X)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T  # binary case safety

    scores = scores[0]  # (n_classes,)
    k = min(top_k, scores.shape[0])
    top_idx = np.argsort(scores)[-k:][::-1]

    top = []
    for idx in top_idx:
        rome = le.inverse_transform([idx])[0]
        top.append({"rome_code": str(rome), "rome_label": rome_index.get(str(rome)), "score": float(scores[idx])})

    return {"rome_pred": top[0]["rome_code"], "rome_label": rome_index.get(top[0]["rome_code"]), "top_k": top}


def main():
    print(f" predict_model — backend={STORAGE_BACKEND}")
    artifacts = load_artifacts()
    print(f"✅ Loaded model: {MODEL_NAME} / {artifacts['version']}")

    # Example payload
    example = {
        "intitule": "Data Engineer",
        "description": "Développement de pipelines de données avec Python, Spark, Airflow. "
                       "Mise en place de jobs ETL, ingestion, stockage sur S3/MinIO.",
        "competences": ["Python", "Spark", "Airflow", "SQL", "S3"],
    }

    text = build_text_payload(
        intitule=example.get("intitule"),
        description=example.get("description"),
        competences=example.get("competences"),
    )

    pred = predict_top_k(artifacts, text, top_k=TOP_K, rome_index=rome_model)
    print("🎯 Prediction:")
    print(json.dumps(pred, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
