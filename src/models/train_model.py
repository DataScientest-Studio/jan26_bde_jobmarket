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

    Hyperparameter Tuning Strategies
    ----------------------
    The pipeline supports 4 tuning strategies (configurable via TUNING_STRATEGY env var):

        1. "none" (default): No tuning, uses default hyperparameters from environment variables.
           Fast baseline for prototyping. Validation set used only for monitoring.

        2. "manual": Explicit loops testing predefined parameter combinations.
           Full control, useful for understanding parameter impact.
           Validation set used to select best configuration.

        3. "grid": GridSearchCV with exhaustive search over a parameter grid.
           Cross-validation on train set, parallel execution.
           More robust but computationally expensive.

        4. "random": RandomizedSearchCV sampling N random combinations.
           Faster than grid search, explores continuous distributions.
           Good for large parameter spaces.

    See HYPERPARAMETER_TUNING.md for detailed usage and examples.


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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

# joblib is used to save and load trained ML artifacts (model, vectorizer, label encoder)
# efficiently, especially for large scikit-learn objects
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from scipy.stats import uniform, randint

from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)

from src.storage.storage import LocalStorage, S3Storage, get_storage_from_env

# -----------------------------
# CONFIG (env overridable)
# -----------------------------
DATASET_KEY = os.getenv("DATASET_KEY", "datasets/rome_dataset.parquet")

MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
MODEL_PATH_PREFIX = os.getenv("MODEL_PATH_PREFIX", "models")  # Relative path within gold layer
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
MODEL_BASE_KEY = f"{MODEL_PATH_PREFIX}/{MODEL_NAME}/versions/{MODEL_VERSION}"

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

# Tuning strategy: "none", "manual", "grid", "random"
TUNING_STRATEGY = os.getenv("TUNING_STRATEGY", "none")
TUNING_CV_FOLDS = int(os.getenv("TUNING_CV_FOLDS", "3"))
TUNING_N_JOBS = int(os.getenv("TUNING_N_JOBS", "-1"))  # -1 = use all CPUs

# Manual tuning parameter grids (comma-separated strings parsed to lists)
MANUAL_TUNING_C_VALUES = [float(x.strip()) for x in os.getenv("MANUAL_TUNING_C_VALUES", "0.1,0.5,1.0,5.0,10.0").split(",")]
MANUAL_TUNING_NGRAM_MAX_VALUES = [int(x.strip()) for x in os.getenv("MANUAL_TUNING_NGRAM_MAX_VALUES", "1,2").split(",")]
MANUAL_TUNING_MIN_DF_VALUES = [int(x.strip()) for x in os.getenv("MANUAL_TUNING_MIN_DF_VALUES", "2,3,5").split(",")]

# Grid Search parameter grids (comma-separated strings parsed to lists)
GRID_SEARCH_NGRAM_RANGE = os.getenv("GRID_SEARCH_NGRAM_RANGE", "1-1,1-2")  # Format: "1-1,1-2,1-3"
GRID_SEARCH_MIN_DF = [int(x.strip()) for x in os.getenv("GRID_SEARCH_MIN_DF", "2,3,5").split(",")]
GRID_SEARCH_MAX_DF = [float(x.strip()) for x in os.getenv("GRID_SEARCH_MAX_DF", "0.85,0.90").split(",")]
GRID_SEARCH_C = [float(x.strip()) for x in os.getenv("GRID_SEARCH_C", "0.1,1.0,10.0").split(",")]

# Storage backends
storage_gold = get_storage_from_env("gold")  # For datasets and models


# Helpers: Config Parsing
# =============================

def parse_ngram_range_string(ngram_str: str) -> list:
    """
    Parse ngram range string like "1-1,1-2,1-3" to list of tuples [(1,1), (1,2), (1,3)]
    """
    ranges = []
    for part in ngram_str.split(","):
        part = part.strip()
        min_n, max_n = map(int, part.split("-"))
        ranges.append((min_n, max_n))
    return ranges

# Helpers: Storage Read/Write
# =============================

def read_parquet(key: str) -> pd.DataFrame:
    """Read parquet from gold storage"""
    return storage_gold.read_parquet(key)

def read_json(key: str) -> Dict[str, Any]:
    """Read JSON from gold storage (for models)"""
    return json.loads(storage_gold.read_bytes(key).decode("utf-8"))

def write_json(key: str, payload: Dict[str, Any]) -> None:
    """Write JSON to gold storage (for models)"""
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    storage_gold.write_bytes(key, data, content_type="application/json; charset=utf-8")

def write_joblib(key: str, obj: Any) -> None:
    """Write joblib to gold storage (for models)"""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    storage_gold.write_bytes(key, buf.getvalue(), content_type="application/octet-stream")

def write_parquet(storage, key: str, df) -> None:
    """Write parquet to specified storage"""
    storage.write_parquet(key, df)

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
# Tuning Strategies
# -----------------------------

def train_without_tuning(X_train, y_train, X_val, y_val):
    """
    Strategy : No hyperparameter tuning (default values from env)
    Returns: (vectorizer, model, config_dict)
    """
    print("📌 Strategy: NO TUNING (using default hyperparameters)")
    
    max_features = int(TFIDF_MAX_FEATURES) if TFIDF_MAX_FEATURES else None
    vectorizer = TfidfVectorizer(
        ngram_range=(TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX),
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        max_features=max_features,
        dtype=np.float32,
    )
    
    print("🧠 Fitting TF-IDF...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    model = LinearSVC(C=SVC_C, random_state=RANDOM_STATE)
    print("🏋️ Training LinearSVC...")
    model.fit(X_train_vec, y_train)
    
    # Validation score (for monitoring)
    val_acc = accuracy_score(y_val, model.predict(X_val_vec))
    print(f"✅ Validation accuracy: {val_acc:.4f}")
    
    config = {
        "strategy": "none",
        "tfidf": {
            "ngram_range": [TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX],
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "sublinear_tf": TFIDF_SUBLINEAR_TF,
            "max_features": max_features,
        },
        "model": {"type": "LinearSVC", "C": SVC_C},
    }
    
    return vectorizer, model, config


def train_with_manual_tuning(X_train, y_train, X_val, y_val):
    """
    Strategy : Manual hyperparameter tuning with explicit loops
    Uses parameter grids from environment variables (MANUAL_TUNING_*)
    Returns: (vectorizer, model, config_dict)
    """
    print("📌 Strategy: MANUAL TUNING")
    
    # Use environment variables for parameter grids
    C_values = MANUAL_TUNING_C_VALUES
    ngram_max_values = MANUAL_TUNING_NGRAM_MAX_VALUES
    min_df_values = MANUAL_TUNING_MIN_DF_VALUES
    
    print(f"  Parameters from env:")
    print(f"    C_values: {C_values}")
    print(f"    ngram_max_values: {ngram_max_values}")
    print(f"    min_df_values: {min_df_values}")
    
    best_val_acc = 0
    best_config = None
    best_model = None
    best_vectorizer = None
    
    total_combinations = len(C_values) * len(ngram_max_values) * len(min_df_values)
    print(f"🔍 Testing {total_combinations} combinations...")
    
    combination_idx = 0
    for C in C_values:
        for ngram_max in ngram_max_values:
            for min_df in min_df_values:
                combination_idx += 1
                
                # Create vectorizer with these parameters
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    max_df=TFIDF_MAX_DF,
                    sublinear_tf=TFIDF_SUBLINEAR_TF,
                    dtype=np.float32,
                )
                
                # Fit on TRAIN, transform train and val
                X_train_vec = vectorizer.fit_transform(X_train)
                X_val_vec = vectorizer.transform(X_val)
                
                # Train model
                model = LinearSVC(C=C, random_state=RANDOM_STATE)
                model.fit(X_train_vec, y_train)
                
                # Evaluate on validation
                y_val_pred = model.predict(X_val_vec)
                val_acc = accuracy_score(y_val, y_val_pred)
                
                print(f"  [{combination_idx}/{total_combinations}] C={C}, ngram_max={ngram_max}, min_df={min_df} → val_acc={val_acc:.4f}")
                
                # Keep best configuration
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = {
                        "C": C,
                        "ngram_max": ngram_max,
                        "min_df": min_df,
                    }
                    best_model = model
                    best_vectorizer = vectorizer
    
    print(f"\n🏆 Best config: {best_config} with val_acc={best_val_acc:.4f}")
    
    config = {
        "strategy": "manual",
        "best_params": best_config,
        "total_combinations_tested": total_combinations,
        "best_val_accuracy": float(best_val_acc),
        "tfidf": {
            "ngram_range": [1, best_config["ngram_max"]],
            "min_df": best_config["min_df"],
            "max_df": TFIDF_MAX_DF,
            "sublinear_tf": TFIDF_SUBLINEAR_TF,
        },
        "model": {"type": "LinearSVC", "C": best_config["C"]},
    }
    
    return best_vectorizer, best_model, config


def train_with_grid_search(X_train, y_train, X_val, y_val):
    """
    Strategy : GridSearchCV with cross-validation
    Uses parameter grids from environment variables (GRID_SEARCH_*)
    Returns: (vectorizer, model, config_dict)
    """
    print("📌 Strategy: GRID SEARCH with Cross-Validation")
    
    # Parse ngram ranges from env
    ngram_ranges = parse_ngram_range_string(GRID_SEARCH_NGRAM_RANGE)
    
    print(f"  Parameters from env:")
    print(f"    ngram_ranges: {ngram_ranges}")
    print(f"    min_df: {GRID_SEARCH_MIN_DF}")
    print(f"    max_df: {GRID_SEARCH_MAX_DF}")
    print(f"    C: {GRID_SEARCH_C}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(dtype=np.float32)),
        ('svc', LinearSVC(random_state=RANDOM_STATE))
    ])
    
    # Define parameter grid using env variables
    param_grid = {
        'tfidf__ngram_range': ngram_ranges,
        'tfidf__min_df': GRID_SEARCH_MIN_DF,
        'tfidf__max_df': GRID_SEARCH_MAX_DF,
        'tfidf__sublinear_tf': [True, False],
        'svc__C': GRID_SEARCH_C
    }
    
    total_combinations = (
        len(param_grid['tfidf__ngram_range']) *
        len(param_grid['tfidf__min_df']) *
        len(param_grid['tfidf__max_df']) *
        len(param_grid['tfidf__sublinear_tf']) *
        len(param_grid['svc__C'])
    )
    print(f"🔍 Testing {total_combinations} combinations with {TUNING_CV_FOLDS}-fold CV...")
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=TUNING_CV_FOLDS,
        scoring='accuracy',
        verbose=2,
        n_jobs=TUNING_N_JOBS
    )
    
    # Fit on TRAIN (GridSearch does CV internally)
    print("🏋️ Running Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n🏆 Best params: {grid_search.best_params_}")
    print(f"📊 Best CV score: {grid_search.best_score_:.4f}")
    
    # Get best model (already trained on full train set)
    best_pipeline = grid_search.best_estimator_
    
    # Validation score
    val_acc = best_pipeline.score(X_val, y_val)
    print(f"✅ Validation accuracy: {val_acc:.4f}")
    
    # Extract vectorizer and model from pipeline
    best_vectorizer = best_pipeline.named_steps['tfidf']
    best_model = best_pipeline.named_steps['svc']
    
    config = {
        "strategy": "grid_search",
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "best_val_accuracy": float(val_acc),
        "cv_folds": TUNING_CV_FOLDS,
        "total_combinations_tested": total_combinations,
        "tfidf": {
            "ngram_range": list(grid_search.best_params_['tfidf__ngram_range']),
            "min_df": grid_search.best_params_['tfidf__min_df'],
            "max_df": grid_search.best_params_['tfidf__max_df'],
            "sublinear_tf": grid_search.best_params_['tfidf__sublinear_tf'],
        },
        "model": {"type": "LinearSVC", "C": grid_search.best_params_['svc__C']},
    }
    
    return best_vectorizer, best_model, config


def train_with_random_search(X_train, y_train, X_val, y_val):
    """
    Strategy : RandomizedSearchCV with cross-validation

    RandomizedSearchCV consist in sampling N random combinations of parameters 
    from specified distributions or lists,and evaluating them with cross-validation. 


    Returns: (vectorizer, model, config_dict)
    """
    print("📌 Strategy: RANDOMIZED SEARCH with Cross-Validation")
    
    # Use a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(dtype=np.float32)),
        ('svc', LinearSVC(random_state=RANDOM_STATE))
    ])
    
    # Define parameter distributions
    param_distributions = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': randint(2, 10),  # Random int between 2 and 10
        'tfidf__max_df': uniform(0.80, 0.15),  # Random float between 0.80 and 0.95 - ex 0.90 ignore terms present in more than 90% of documents
        'tfidf__sublinear_tf': [True, False], # sublinear_tf is a boolean, so we can just list the options
        'svc__C': uniform(0.1, 10)  # Random float between 0.1 and 10.1
    }
    
    n_iter = 20  # Number of random combinations to test
    print(f"🔍 Testing {n_iter} random combinations with {TUNING_CV_FOLDS}-fold CV...")
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=TUNING_CV_FOLDS,
        scoring='accuracy',
        verbose=2,
        n_jobs=TUNING_N_JOBS,
        random_state=RANDOM_STATE
    )
    
    # Fit on TRAIN (RandomSearch does CV internally)
    print("🏋️ Running Randomized Search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n🏆 Best params: {random_search.best_params_}")
    print(f"📊 Best CV score: {random_search.best_score_:.4f}")
    
    # Get best model
    best_pipeline = random_search.best_estimator_
    
    # Validation score
    val_acc = best_pipeline.score(X_val, y_val)
    print(f"✅ Validation accuracy: {val_acc:.4f}")
    
    # Extract vectorizer and model from pipeline
    best_vectorizer = best_pipeline.named_steps['tfidf']
    best_model = best_pipeline.named_steps['svc']
    
    config = {
        "strategy": "random_search",
        "best_params": random_search.best_params_,
        "best_cv_score": float(random_search.best_score_),
        "best_val_accuracy": float(val_acc),
        "cv_folds": TUNING_CV_FOLDS,
        "n_iter": n_iter,
        "tfidf": {
            "ngram_range": list(random_search.best_params_['tfidf__ngram_range']),
            "min_df": int(random_search.best_params_['tfidf__min_df']),
            "max_df": float(random_search.best_params_['tfidf__max_df']),
            "sublinear_tf": random_search.best_params_['tfidf__sublinear_tf'],
        },
        "model": {"type": "LinearSVC", "C": float(random_search.best_params_['svc__C'])},
    }
    
    return best_vectorizer, best_model, config


def get_tuning_strategy(strategy_name: str):
    """
    Factory function to get the tuning fucntion based on strategy name.

    Parameters:
    - strategy_name: str, one of "none", "manual", "grid", "random"

    Returns :
        - "none": train_without_tuning
        - "manual": train_with_manual_tuning
        - "grid": train_with_grid_search
        - "random": train_with_random_search
    """
    strategies = {
        "none": train_without_tuning,
        "manual": train_with_manual_tuning,
        "grid": train_with_grid_search,
        "random": train_with_random_search,
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown tuning strategy: {strategy_name}. "
            f"Available strategies: {list(strategies.keys())}"
        )
    
    return strategies[strategy_name]


# -----------------------------
# Training
# -----------------------------
def main():
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f" train.py — start (backend={os.getenv('STORAGE_BACKEND', 'local')})")
    print(f"📥 Reading dataset: {DATASET_KEY}")

    df = read_parquet(DATASET_KEY)

    required_cols = {"text", "romeCode"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Dataset must contain columns {required_cols}, found: {list(df.columns)}")

    # Drop rows with missing text or labels, and reset index for clean slicing later.
    df = df.dropna(subset=["text", "romeCode"]).reset_index(drop=True)
    print(f"📊 Dataset rows: {len(df)} | classes: {df['romeCode'].nunique()}")

    # X is the text content (features)
    X = df["text"].astype(str).values
    # y_raw is the original ROME code labels (targets) before encoding.
    y_raw = df["romeCode"].astype(str).values

    # Encode labels with LabelEncoder. This converts string labels to integers
    #  (0..n_classes-1) to avoid colinearity issues and ensure compatibility with scikit-learn classifiers.
    le = LabelEncoder()
    # y will be the encoded integer labels corresponding to the original ROME codes in y_raw.
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"🏷️ Classes encoded: {n_classes}")

    # Convert X and y to NumPy arrays to ensure compatibility with train_test_split.
    # scikit-learn uses NumPy-based indexing internally, and Python lists
    # do not support indexing with NumPy arrays, which can raise a TypeError.
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)

    # Split: train set +validation set / test set
    # Stratify by y to maintain class distribution across splits. 
    # Use random_state for reproducibility.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, 
        y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    # Then split train_val in to : train set  + validation set 
    # validation set is used for tuning hyperparameters model, while test set is only for final evaluation.
    val_ratio_on_trainval = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, 
        y_trainval, 
        test_size=val_ratio_on_trainval, 
        random_state=RANDOM_STATE, 
        stratify=y_trainval
    )

    print(f"✂️ Split sizes: train set ={len(X_train)} | validation set ={len(X_val)} | test set ={len(X_test)}")

    # ==============================
    # HYPERPARAMETER TUNING
    # ==============================
    print(f"\n{'='*60}")
    print(f"TUNING STRATEGY: {TUNING_STRATEGY.upper()}")
    print(f"{'='*60}\n")
    
    # Select and execute tuning strategy
    training_start_time = time.time()
    tuning_function = get_tuning_strategy(TUNING_STRATEGY)
    vectorizer, model, tuning_config = tuning_function(X_train, y_train, X_val, y_val)
    training_end_time = time.time()
    training_duration_seconds = training_end_time - training_start_time
    
    # ==============================
    # FINAL EVALUATION
    # ==============================
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON ALL SPLITS")
    print(f"{'='*60}\n")
    
    # Transform all splits with the final vectorizer
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # Evaluate on all splits
    def eval_split(name: str, Xv, yv) -> Dict[str, float]:
        y_pred = model.predict(Xv)
        acc = accuracy_score(yv, y_pred)
        f1m = f1_score(yv, y_pred, average="macro")

        # decision_function for top-k
        scores = model.decision_function(Xv)

        # In multiclass, shape : (n_samples, n_classes). 
        # In binary classification : (n_samples,)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T

        top3 = top_k_accuracy_from_scores(scores, yv, k=min(3, scores.shape[1]))
        top5 = top_k_accuracy_from_scores(scores, yv, k=min(5, scores.shape[1]))

        print(f"📈 {name}: acc={acc:.4f} | f1_macro={f1m:.4f} | top3={top3:.4f} | top5={top5:.4f}")
        return {"accuracy": float(acc), "f1_macro": float(f1m), "top3": float(top3), "top5": float(top5)}

    # Calculate metrics on train, validation, and test sets.
    metrics = {
        "train": eval_split("train", X_train_vec, y_train),
        "val": eval_split("val", X_val_vec, y_val),
        "test": eval_split("test", X_test_vec, y_test),
    }

    # ==============================
    # SAVE ARTIFACTS
    # ==============================
    print(f"\n💾 Saving artifacts to: {MODEL_BASE_KEY}/")
    write_joblib(f"{MODEL_BASE_KEY}/vectorizer.joblib", vectorizer)
    write_joblib(f"{MODEL_BASE_KEY}/model.joblib", model)
    write_joblib(f"{MODEL_BASE_KEY}/label_encoder.joblib", le)

    # Merge tuning config with base config
    config = {
        "dataset_key": DATASET_KEY,
        "tuning": tuning_config,
        "split": {
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
        },
        "labels": {"n_classes": n_classes},
    }

    train_meta = {
        "run_ts_utc": run_ts,
        "backend": os.getenv('STORAGE_BACKEND', 'local'),
        "rows": int(len(df)),
        "classes": int(n_classes),
        "tuning_strategy": TUNING_STRATEGY,
        "training_duration_seconds": float(training_duration_seconds),
        "training_duration_minutes": float(training_duration_seconds / 60),
    }

    write_json(f"{MODEL_BASE_KEY}/metrics.json", metrics)
    write_json(f"{MODEL_BASE_KEY}/config.json", config)
    write_json(f"{MODEL_BASE_KEY}/train_meta.json", train_meta)

    # Optional: also write a "latest" pointer (simple file) – works for local & S3
    # This avoids symlinks (Windows/S3).
    write_json(f"{MODEL_PATH_PREFIX}/{MODEL_NAME}/LATEST.json", {"latest": MODEL_VERSION, "updated_utc": run_ts})

    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Strategy: {TUNING_STRATEGY}")
    print(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test F1-Macro: {metrics['test']['f1_macro']:.4f}")
    print(f"Artifacts saved to: {MODEL_BASE_KEY}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
