"""
This module implements an end-to-end text classification pipeline to predict a France Travail ROME code 
from the textual content of a job offer.

Pipeline overview :

    Ingestion and storage
    ----------------------
    Job offers are collected from the France Travail API using pagination. 
    The API limits responses to 150 results per request, so the ingestion logic iterates over successive 
    range windows until the maximum available range is reached. 
    Raw offers are stored in the Bronze layer as JSONL (NDJSON) files. 
    JSONL is well-suited for ingestion because each record is independent, 
    files can be written as immutable parts (part-000001.jsonl, part-000002.jsonl), 
    and this pattern is compatible with object storage systems such as S3/MinIO 
    which do not support efficient append operations.

    Dataset creation (Gold)
    ----------------------
    The dataset builder reads Bronze JSONL files and constructs an ML-ready table 
    with a single text field and a target label. 
    The text field concatenates relevant sections such as title, description, 
    and (when available) structured competences. 

    The output is written to the Gold layer as Parquet (columnar storage), 
    which is smaller and faster to read than JSONL for training and analytics.

    Target encoding (ROME codes)
    ----------------------
    ROME codes are categorical strings (e.g., "M1805", "C1504"). 
    Since scikit-learn classifiers operate on numeric targets, 
    labels are encoded into integer indices using a LabelEncoder. 
    
    This creates a stable mapping such as "M1805" -> 0, "C1504" -> 1, etc. 
    The integer encoding does not imply any ordinal relationship; 
    it is only a technical requirement for multi-class classification.

    Feature engineering with TF-IDF and sparse matrices
    ----------------------
    Text is transformed into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency). 
    With a large corpus and n-grams (unigrams/bigrams), 
    the vocabulary can reach hundreds of thousands of features. 

    The resulting design matrix has shape approximately (n_samples, n_features), 
    for example (307,000 x 200,000). 
    It is sparse because each document contains only a small fraction of the global vocabulary, 
    so most feature values are zero. 
    
    For efficiency, the matrix is stored in a compressed sparse format (CSR), 
    which keeps only non-zero values and their indices, 
    reducing memory usage and enabling fast linear operations.

    Model choice: LinearSVC (linear SVM) versus Logistic Regression
    ----------------------
    The chosen model is LinearSVC, a linear Support Vector Machine trained 
    to separate classes with hyperplanes in the high-dimensional TF-IDF space. 
    
    Text classification with TF-IDF often becomes close to linearly separable 
    because the representation is high-dimensional and sparse, making a linear decision boundary effective. 
    
    LinearSVC is typically efficient and robust on large sparse matrices and multi-class setups, 
    and it avoids the heavier computational costs that can appear with Logistic Regression 
    when the number of classes and features is large (solver sensitivity, higher memory/CPU requirements). 
    
    Top-k predictions can be produced by ranking the decision scores returned by the model's decision function.

    Why not a Transformer (BERT) model
    ----------------------
    Transformer-based approaches can capture deeper semantic relationships, 
    but they come with higher operational cost: GPU requirements (or much slower CPU inference), 
    increased memory usage, longer training times, and additional complexity for fine-tuning and deployment. 
    
    In a large-scale, structured job-offer domain where discriminative keywords and phrases are strong signals, 
    TF-IDF plus a linear classifier is a strong and lightweight baseline 
    that is easier to train, version, and deploy in a Dockerized environment. 
    
    A Transformer becomes more justified when semantic nuance is critical, labeled data is limited, 
    and the infrastructure can support deep learning workflows.

    Evaluation metrics and their usage
    ----------------------
    Multiple metrics are used due to the large number of classes and potential class imbalance:

        - Accuracy measures overall correctness (percentage of exact matches) 
        but can be misleading if classes are imbalanced.

        - Macro F1-score computes F1 per class and averages equally across classes, 
        providing a fairer view of performance on minority classes. 
        
        F1 per class explains which ROME codes are well-predicted and which are not.
        So it is crucial for diagnosing model performance across the diverse set of ROME codes.
        
        - Top-k accuracy (e.g., top-3, top-5) measures whether the true ROME code 
        appears among the k highest-scoring predictions. 


                        ┌─────────────────────────┐
                        │ FT Bronze Layer (JSONL) │
                        │ partitionné :           │
                        │ dt=YYYY-MM-DD           │
                        │ run_id=timestamp        │
                        │ part-000001.jsonl       │
                        └─────────────┬───────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │  Dataset Builder        │
                        │  make_dataset.py        │
                        │  - Nettoyage texte      │
                        │  - Construction champ ML│
                        │  - Filtrage classes     │
                        └─────────────┬───────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │ Gold Layer (Parquet)    │
                        │ rome_dataset.parquet    │
                        └─────────────┬───────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   Training Script       │
                        │   train_model.py        │
                        │                         │
                        │   1. Split 80/10/10     │
                        │   2. TF-IDF Vectorizer  │
                        │   3. LinearSVC          │
                        │   4. Metrics (Top-K)    │
                        └─────────────┬───────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │  Model Artifacts        │
                        │  models/rome_tfidf/     │
                        │    versions/vX/         │
                        │      model.joblib       │
                        │      vectorizer.joblib  │
                        │      metrics.json       │
                        │    LATEST.json          │
                        └─────────────────────────┘


"""




import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any
import boto3

# joblib is used to save and load trained ML artifacts (model, vectorizer, label encoder)
# efficiently, especially for large scikit-learn objects
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)

# -----------------------------
# CONFIG (env overridable)
# -----------------------------
DATASET_KEY = os.getenv("DATASET_KEY", "jobmarket/gold/datasets/rome_dataset.parquet")

MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_BASE_KEY = f"models/{MODEL_NAME}/versions/{MODEL_VERSION}"

# Split ratios
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.10"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.10"))  # applied on remaining train after test split
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# TF-IDF params (bons défauts)
TFIDF_NGRAM_MIN = int(os.getenv("TFIDF_NGRAM_MIN", "1"))
TFIDF_NGRAM_MAX = int(os.getenv("TFIDF_NGRAM_MAX", "2"))
TFIDF_MIN_DF = int(os.getenv("TFIDF_MIN_DF", "3"))
TFIDF_MAX_DF = float(os.getenv("TFIDF_MAX_DF", "0.90"))
TFIDF_SUBLINEAR_TF = os.getenv("TFIDF_SUBLINEAR_TF", "true").lower() == "true"
TFIDF_MAX_FEATURES = os.getenv("TFIDF_MAX_FEATURES")  # None or int

# LinearSVC params
SVC_C = float(os.getenv("SVC_C", "1.0"))

# I/O backend (same logic as your storage.py)
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower().strip()
FT_DATA_DIR = Path(os.getenv("FT_DATA_DIR", "data/france_travail"))

# S3 / MinIO env
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = (os.getenv("S3_PREFIX_FT", "") or "").strip("/")
S3_REGION = os.getenv("S3_REGION", "us-east-1")


# -----------------------------
# Helpers: S3
# -----------------------------
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
    if S3_PREFIX:
        return f"{S3_PREFIX}/{normalized}"
    return normalized

def _require_s3_env():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is required when STORAGE_BACKEND=s3")


# -----------------------------
# Read dataset (local or S3)
# -----------------------------
def read_parquet(key: str) -> pd.DataFrame:
    if STORAGE_BACKEND == "local":
        path = FT_DATA_DIR / Path(key)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return pd.read_parquet(path)

    if STORAGE_BACKEND == "s3":
        _require_s3_env()
        client = _s3_client()
        full_key = _s3_full_key(key)
        obj = client.get_object(Bucket=S3_BUCKET, Key=full_key)
        data = obj["Body"].read()
        buf = io.BytesIO(data)
        return pd.read_parquet(buf)

    raise RuntimeError(f"Unsupported STORAGE_BACKEND={STORAGE_BACKEND}")


# -----------------------------
# Write artifacts (local or S3)
# -----------------------------
def write_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    if STORAGE_BACKEND == "local":
        out_path = FT_DATA_DIR / Path(key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return

    if STORAGE_BACKEND == "s3":
        _require_s3_env()
        client = _s3_client()
        full_key = _s3_full_key(key)
        client.put_object(
            Bucket=S3_BUCKET,
            Key=full_key,
            Body=data,
            ContentType=content_type,
        )
        return

    raise RuntimeError(f"Unsupported STORAGE_BACKEND={STORAGE_BACKEND}")


def write_json(key: str, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    write_bytes(key, data, content_type="application/json; charset=utf-8")


def write_joblib(key: str, obj: Any) -> None:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    write_bytes(key, buf.getvalue(), content_type="application/octet-stream")


# -----------------------------
# Metrics: top-k accuracy for LinearSVC
# -----------------------------
def top_k_accuracy_from_scores(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """
    scores: shape (n_samples, n_classes)
    y_true: encoded labels (0..n_classes-1)
    """
    # argsort descending and take top-k indices
    topk = np.argsort(scores, axis=1)[:, -k:]
    # check if true label is in topk
    hits = np.any(topk == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(hits))


# -----------------------------
# Training
# -----------------------------
def main():
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f" train.py — start (backend={STORAGE_BACKEND})")
    print(f"📥 Reading dataset: {DATASET_KEY}")

    df = read_parquet(DATASET_KEY)

    required_cols = {"text", "romeCode"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Dataset must contain columns {required_cols}, found: {list(df.columns)}")

    # Drop rows with missing text or labels, and reset index for clean slicing later.
    df = df.dropna(subset=["text", "romeCode"]).reset_index(drop=True)
    print(f"📊 Dataset rows: {len(df)} | classes: {df['romeCode'].nunique()}")

    X = df["text"].astype(str).values
    # y_raw is the original string labels (ROME codes) before encoding
    y_raw = df["romeCode"].astype(str).values

    # Encode labels with LabelEncoder. This converts string labels to integers (0..n_classes-1).
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"🏷️ Classes encoded: {n_classes}")

    # Convert X and y to NumPy arrays to ensure compatibility with train_test_split.
    # scikit-learn uses NumPy-based indexing internally, and Python lists
    # do not support indexing with NumPy arrays, which can raise a TypeError.
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)

    # Split: train set +validation set / test set
    # Stratify by y to maintain class distribution across splits. Use random_state for reproducibility.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Then split train_val in to : train set  + validation set 
    val_ratio_on_trainval = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_on_trainval, random_state=RANDOM_STATE, stratify=y_trainval
    )

    print(f"✂️ Split sizes: train set ={len(X_train)} | validation set ={len(X_val)} | test set ={len(X_test)}")

    # TF-IDF
    max_features = int(TFIDF_MAX_FEATURES) if TFIDF_MAX_FEATURES else None
    vectorizer = TfidfVectorizer(
        ngram_range=(TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX),
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        max_features=max_features,
        dtype=np.float32,   # to reduce memory usage vs default float64
    )

    print("🧠 Fitting TF-IDF…")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # LinearSVC is efficient and well-suited for large sparse TF-IDF feature spaces.
    # We are working with a large sparse TF-IDF feature space:
    # - "Large" because the vocabulary (unigrams + bigrams) across ~300k job offers
    #   can generate 150k to 300k features.
    # - "Sparse" because each individual document only contains a small fraction
    #   of the total vocabulary, meaning most feature values are zero.
    # This results in a high-dimensional sparse matrix, which LinearSVC
    # handles efficiently and is well-suited for text classification tasks.    
    model = LinearSVC(C=SVC_C)

    print("🏋️ Training LinearSVC…")
    model.fit(X_train_vec, y_train)

    # Evaluate
    def eval_split(name: str, Xv, yv) -> Dict[str, float]:
        y_pred = model.predict(Xv)
        acc = accuracy_score(yv, y_pred)
        f1m = f1_score(yv, y_pred, average="macro")

        # decision_function for top-k
        scores = model.decision_function(Xv)
        # In multiclass, shape = (n_samples, n_classes). In binary, it's (n_samples,)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T

        top3 = top_k_accuracy_from_scores(scores, yv, k=min(3, scores.shape[1]))
        top5 = top_k_accuracy_from_scores(scores, yv, k=min(5, scores.shape[1]))

        print(f"📈 {name}: acc={acc:.4f} | f1_macro={f1m:.4f} | top3={top3:.4f} | top5={top5:.4f}")
        return {"accuracy": float(acc), "f1_macro": float(f1m), "top3": float(top3), "top5": float(top5)}

    metrics = {
        "train": eval_split("train", X_train_vec, y_train),
        "val": eval_split("val", X_val_vec, y_val),
        "test": eval_split("test", X_test_vec, y_test),
    }

    # Save artifacts
    print(f"💾 Saving artifacts to: {MODEL_BASE_KEY}/")
    write_joblib(f"{MODEL_BASE_KEY}/vectorizer.joblib", vectorizer)
    write_joblib(f"{MODEL_BASE_KEY}/model.joblib", model)
    write_joblib(f"{MODEL_BASE_KEY}/label_encoder.joblib", le)

    config = {
        "": DATASET_KEY,
        "tfidf": {
            "ngram_range": [TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX],
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "sublinear_tf": TFIDF_SUBLINEAR_TF,
            "max_features": max_features,
        },
        "model": {"type": "LinearSVC", "C": SVC_C},
        "split": {
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
        },
        "labels": {"n_classes": n_classes},
    }

    train_meta = {
        "run_ts_utc": run_ts,
        "backend": STORAGE_BACKEND,
        "rows": int(len(df)),
        "classes": int(n_classes),
    }

    write_json(f"{MODEL_BASE_KEY}/metrics.json", metrics)
    write_json(f"{MODEL_BASE_KEY}/config.json", config)
    write_json(f"{MODEL_BASE_KEY}/train_meta.json", train_meta)

    # Optional: also write a "latest" pointer (simple file) – works for local & S3
    # This avoids symlinks (Windows/S3).
    write_json(f"models/{MODEL_NAME}/LATEST.json", {"latest": MODEL_VERSION, "updated_utc": run_ts})

    print("✅ train.py — done")


if __name__ == "__main__":
    main()
