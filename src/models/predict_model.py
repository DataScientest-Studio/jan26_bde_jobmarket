import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import joblib
import numpy as np

from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
TOP_K = int(os.getenv("TOP_K", "5"))

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower().strip()
FT_DATA_DIR = Path(os.getenv("FT_DATA_DIR", "data/france_travail"))

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = (os.getenv("S3_PREFIX_FT", "") or "").strip("/")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

# -----------------------------
# Helpers: S3
# -----------------------------
def _require_s3_env():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is required when STORAGE_BACKEND=s3")


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    )


def _s3_full_key(key: str) -> str:
    normalized = key.lstrip("/").replace("\\", "/")
    return f"{S3_PREFIX}/{normalized}" if S3_PREFIX else normalized



def read_json(key: str) -> Dict[str, Any]:
    return json.loads(read_bytes(key).decode("utf-8"))

def load_joblib(key: str) -> Any:
    data = read_bytes(key)
    return joblib.load(io.BytesIO(data))


# -----------------------------
# Get ROME model
# -----------------------------
def read_bytes(key: str) -> bytes:
    if STORAGE_BACKEND == "local":
        path = FT_DATA_DIR / Path(key)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_bytes()

    if STORAGE_BACKEND == "s3":
        _require_s3_env()
        client = _s3_client()
        obj = client.get_object(Bucket=S3_BUCKET, Key=_s3_full_key(key))
        return obj["Body"].read()

    raise RuntimeError(f"Unsupported STORAGE_BACKEND={STORAGE_BACKEND}")


def read_jsonl(key: str) -> List[Dict[str, Any]]:
    """ Read JSONL file and return list of dicts """
    results = []
    data = read_bytes(key).decode("utf-8")
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        results.append(json.loads(line))
    return results

def get_rome_model() -> str:
    rome_model = read_jsonl(f"bronze/rome/rome_metiers.jsonl")
    rome_index = {row["code"]: row["libelle"] for row in rome_model}
    return rome_index

global rome_model
rome_model = get_rome_model()
print(f"✅ Loaded ROME model: {len(rome_model)} entries")





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
    latest = read_json(f"models/{MODEL_NAME}/LATEST.json")
    return latest["latest"]


def load_artifacts() -> Dict[str, Any]:
    version = get_latest_version()
    base = f"models/{MODEL_NAME}/versions/{version}"
    vectorizer = load_joblib(f"{base}/vectorizer.joblib")
    model = load_joblib(f"{base}/model.joblib")
    label_encoder = load_joblib(f"{base}/label_encoder.joblib")
    return {"version": version, "base": base, "vectorizer": vectorizer, "model": model, "label_encoder": label_encoder}


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
