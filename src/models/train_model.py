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

    Model choice: SGDClassifier(loss='hinge') — same objective as LinearSVC, lower memory
    ----------------------
    The classifier is SGDClassifier(loss='hinge'), which optimises the same hinge loss
    as LinearSVC (linear SVM) and produces equivalent decision boundaries, but uses
    Stochastic Gradient Descent instead of the LIBLINEAR batch solver.

    The key difference is memory complexity:
      - LinearSVC (LIBLINEAR): O(n_samples × n_classes) working memory — with 580 000 rows
        and 992 classes the internal gradient structures saturate a 8 GB container (OOM / SIGKILL).
      - SGDClassifier(hinge): O(n_features) — only the final weight matrix is kept in RAM,
        regardless of dataset size.

    Both models support class_weight="balanced" and decision_function() for top-k ranking.
    Convergence is stochastic with SGD (not guaranteed optimal), but empirically equivalent
    on large TF-IDF sparse matrices where the problem is nearly linearly separable.

    Text classification with TF-IDF often becomes close to linearly separable
    because the representation is high-dimensional and sparse, making a linear decision boundary effective.
    This makes both LinearSVC and SGDClassifier(hinge) strong and lightweight baselines.

    Top-k predictions are produced by ranking the decision scores returned by decision_function().

    Why not Logistic Regression: solver sensitivity and higher memory/CPU requirements when
    the number of classes and features is large.

    See docs/ML.md for a full comparison of LinearSVC vs SGDClassifier.

    Class imbalance and class_weight="balanced"
    ----------------------
    Job offer data is naturally skewed: common occupations (retail, logistics, IT) generate
    10-50x more offers than specialised ones. Without correction, the model optimises the
    global margin and over-represents majority classes — minority classes end up with low
    recall and a degraded macro F1, even if overall accuracy looks acceptable.

    class_weight="balanced" corrects this by reweighting each class inversely proportional
    to its frequency in the training set:

        weight_i = n_samples / (n_classes * count_i)

    A class with 80 examples gets a much higher weight than one with 8 000, forcing the
    model to treat errors on rare classes as more costly. This improves macro F1 without
    modifying the dataset or losing any training data.

    Configurable via SVC_CLASS_WEIGHT env var ("balanced" by default, "none" to disable).
    See docs/ML.md for a detailed explanation of metrics and class imbalance strategies.

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
    Multiple metrics are used due to the large number of classes and potential class imbalance.
    See docs/ML.md for full definitions.

        - Accuracy: overall correctness (% exact matches). Misleading on imbalanced data
          because majority classes dominate the score.

        - Macro F1: F1 computed per class, then averaged equally across all classes.
          Each of the ~490 ROME codes counts for 1/490 regardless of frequency.
          This is the primary indicator — it reveals whether rare occupations are
          correctly covered, not just the common ones.

        - F1 per class = harmonic mean of precision and recall for that class.
          Precision: "when I predict this class, am I right?"
          Recall:    "among all real cases of this class, how many did I find?"

        - Top-k accuracy (e.g., top-3, top-5): whether the true ROME code appears
          among the k highest-scoring predictions. Useful for UX where the user
          can select from a short suggested list.

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

    CLI usage examples
    ----------------------
    # Standard run — latest Gold dt, cap=500 (MAX_CLASS_COUNT from .env), updates LATEST.json
    python -m src.models.train_model

    # Specific dataset date
    python -m src.models.train_model --dt 2026-03-23

    # Experiment run — cap override, no LATEST.json update, versioned path includes cap+dt
    python -m src.models.train_model --max-class-count 480 --no-update-latest

    # Disable capping entirely
    python -m src.models.train_model --max-class-count 0 --no-update-latest

    # With manual tuning strategy
    python -m src.models.train_model --tuning-strategy manual --no-update-latest

    # Full experiment: specific dt, custom cap, no prod update
    python -m src.models.train_model --dt 2026-03-09 --max-class-count 500 --no-update-latest

    Key env vars (configurable in .env):
        MAX_CLASS_COUNT     Cap per class before training (default: 500, 0 = disabled)
        --tuning-strategy   none | manual | grid | random (default: TUNING_STRATEGY env var)
        SVC_CLASS_WEIGHT    balanced | none (default: balanced)
        MODEL_VERSION       Version label prefix (e.g. v2) — combined with dt: v2_cap500_2026-03-23
        TFIDF_MAX_FEATURES  Max TF-IDF vocabulary size (default: 200000)


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
                        │   3. SGDClassifier      │
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
import argparse
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any

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
from sklearn.linear_model import SGDClassifier
from scipy.stats import uniform, randint

from src.config.env import load_project_env
from src.storage.storage import get_storage_from_env

load_project_env()  # safe à rappeler (idempotent)
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG (env overridable)
# -----------------------------
# Dataset key template — dt is resolved at runtime (auto or explicit --dt)
DATASET_KEY_TEMPLATE = "datasets/{dt}/rome_dataset.parquet"

MODEL_NAME = os.getenv("MODEL_NAME", "rome_tfidf")
MODEL_PATH_PREFIX = os.getenv("MODEL_PATH_PREFIX", "models")  # Relative path within gold layer
# Base version label — dt is appended at runtime to form the effective version:
#   effective_version = f"{MODEL_VERSION}_{dt}"  e.g. "v2_2026-02-13" or "cap_p90_2026-02-13"
# MODEL_BASE_KEY is constructed inside train() once dt is known.
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

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

# Class capping — applied at training time, not in the dataset file.
# Undersamples majority classes to at most MAX_CLASS_COUNT examples before training.
# The dataset parquet files remain untouched; the cap is a training hyperparameter.
# Set to 0 or empty to disable capping.
# Recommended: p90 of the class frequency distribution (see docs/ML.md and analyze_dataset.py).
MAX_CLASS_COUNT = int(os.getenv("MAX_CLASS_COUNT", "0"))

# SGDClassifier params (remplace LinearSVC — même objectif hinge loss, mémoire O(n_features) vs O(n_samples))
# SVC_C est conservé comme interface : alpha = 1/(n_train * C) est calculé dynamiquement.
# Voir docs/ML.md pour le détail de la conversion et les raisons du remplacement.
SVC_C = float(os.getenv("SVC_C", "1.0"))
# class_weight="balanced" reweights each class inversely proportional to its frequency.
# This corrects the bias toward majority classes without modifying the dataset.
# Set SVC_CLASS_WEIGHT=none to disable.
SVC_CLASS_WEIGHT = os.getenv("SVC_CLASS_WEIGHT", "balanced") or None

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
    
    # alpha = 1/(n_train * C) — conversion standard LinearSVC C → SGD alpha (voir docs/ML.md)
    n_train = X_train_vec.shape[0]
    sgd_alpha = 1.0 / (n_train * SVC_C)
    model = SGDClassifier(
        loss='hinge', alpha=sgd_alpha, class_weight=SVC_CLASS_WEIGHT,
        max_iter=1000, tol=1e-3, random_state=RANDOM_STATE,
    )
    print("🏋️ Training SGDClassifier(hinge)...")
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
        "model": {"type": "SGDClassifier", "loss": "hinge", "alpha": sgd_alpha, "equivalent_C": SVC_C},
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
    
    print("  Parameters from env:")
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
                
                # Train model — alpha = 1/(n_train * C) (voir docs/ML.md)
                sgd_alpha = 1.0 / (X_train_vec.shape[0] * C)
                model = SGDClassifier(
                    loss='hinge', alpha=sgd_alpha, class_weight=SVC_CLASS_WEIGHT,
                    max_iter=1000, tol=1e-3, random_state=RANDOM_STATE,
                )
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
        "model": {"type": "SGDClassifier", "loss": "hinge", "equivalent_C": best_config["C"]},
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
    
    print("  Parameters from env:")
    print(f"    ngram_ranges: {ngram_ranges}")
    print(f"    min_df: {GRID_SEARCH_MIN_DF}")
    print(f"    max_df: {GRID_SEARCH_MAX_DF}")
    print(f"    C: {GRID_SEARCH_C}")
    
    # Create pipeline — SGDClassifier remplace LinearSVC (voir docs/ML.md)
    # alpha = 1/(n_train * C) : les valeurs C de GRID_SEARCH_C sont converties en alpha.
    n_train = X_train.shape[0]
    alpha_values = [1.0 / (n_train * C) for C in GRID_SEARCH_C]
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(dtype=np.float32)),
        ('svc', SGDClassifier(loss='hinge', class_weight=SVC_CLASS_WEIGHT,
                              max_iter=1000, tol=1e-3, random_state=RANDOM_STATE))
    ])

    # Define parameter grid using env variables
    param_grid = {
        'tfidf__ngram_range': ngram_ranges,
        'tfidf__min_df': GRID_SEARCH_MIN_DF,
        'tfidf__max_df': GRID_SEARCH_MAX_DF,
        'tfidf__sublinear_tf': [True, False],
        'svc__alpha': alpha_values,
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
        "model": {"type": "SGDClassifier", "loss": "hinge", "alpha": grid_search.best_params_['svc__alpha']},
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
    
    # Use a pipeline — SGDClassifier remplace LinearSVC (voir docs/ML.md)
    # alpha = 1/(n_train * C) : C ∈ [0.1, 10.1] → alpha ∈ [1/(n*10.1), 1/(n*0.1)]
    n_train = X_train.shape[0]
    alpha_low = 1.0 / (n_train * 10.1)
    alpha_high = 1.0 / (n_train * 0.1)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(dtype=np.float32)),
        ('svc', SGDClassifier(loss='hinge', class_weight=SVC_CLASS_WEIGHT,
                              max_iter=1000, tol=1e-3, random_state=RANDOM_STATE))
    ])

    # Define parameter distributions
    param_distributions = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': randint(2, 10),
        'tfidf__max_df': uniform(0.80, 0.15),
        'tfidf__sublinear_tf': [True, False],
        'svc__alpha': uniform(alpha_low, alpha_high - alpha_low),
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
        "model": {"type": "SGDClassifier", "loss": "hinge", "alpha": float(random_search.best_params_['svc__alpha'])},
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
# Dataset resolution
# -----------------------------

def resolve_dataset_dt(dt_arg: str | None) -> str:
    """Return the dt to load: explicit value or latest available in Gold datasets.

    Gold datasets are stored as datasets/{YYYY-MM-DD}/rome_dataset.parquet.
    Partitions are plain date folders (not dt= prefixed), so list_prefixes("datasets/")
    returns entries like ["2026-02-13/", "2026-03-07/"]. max() gives the latest.
    """
    if dt_arg and dt_arg not in ("", "latest"):
        return dt_arg
    prefixes = storage_gold.list_prefixes("datasets/")
    dts = [p.strip("/") for p in prefixes if p.strip("/")]
    if not dts:
        raise RuntimeError(
            "No dataset found in Gold. "
            "Run make_dataset or make_dataset_from_silver first."
        )
    latest = max(dts)
    logger.info("Auto mode — latest Gold dataset dt: %s", latest)
    return latest


# -----------------------------
# Training
# -----------------------------

def train(dt: str | None = None, update_latest: bool = True, max_class_count: int | None = None) -> dict:
    """
    Run the full training pipeline for the given dataset partition.

    Loads the Gold dataset at datasets/{dt}/rome_dataset.parquet, splits it into
    train/val/test, applies the configured tuning strategy, evaluates the model,
    and saves artifacts (vectorizer, model, label encoder, metrics, config) to Gold.

    Args:
        dt: Dataset partition date (YYYY-MM-DD). None or "latest" triggers auto mode
            (picks the most recent partition in Gold datasets/).
        update_latest: If True (default), writes LATEST.json to point to this version.
            Set to False for experiment runs (cap tuning, ablations) to avoid
            overwriting the production pointer.

    Returns:
        Dict with dt, dataset_key, rows, classes, metrics, model_base_key,
        effective_version, tuning_strategy, training_duration_seconds.
    """
    dt = resolve_dataset_dt(dt)
    dataset_key = DATASET_KEY_TEMPLATE.format(dt=dt)

    # max_class_count: CLI arg takes precedence over env var (MAX_CLASS_COUNT)
    effective_cap = max_class_count if max_class_count is not None else MAX_CLASS_COUNT

    # Version includes dt and cap so each (MODEL_VERSION, cap, dt) triplet has a unique storage path.
    # Examples: "v1_cap1561_2026-02-13", "v1_nocap_2026-02-13"
    cap_tag = f"cap{effective_cap}" if effective_cap > 0 else "nocap"
    effective_version = f"{MODEL_VERSION}_{cap_tag}_{dt}"
    model_base_key = f"{MODEL_PATH_PREFIX}/{MODEL_NAME}/versions/{effective_version}"

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("train — start (backend=%s)", os.getenv("STORAGE_BACKEND", "local"))
    logger.info("Reading dataset: %s", dataset_key)

    df = read_parquet(dataset_key)

    required_cols = {"text", "rome_code"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Dataset must contain columns {required_cols}, found: {list(df.columns)}")

    # Drop rows with missing text or labels, and reset index for clean slicing later.
    df = df.dropna(subset=["text", "rome_code"]).reset_index(drop=True)
    print(f"📊 Dataset rows: {len(df)} | classes: {df['rome_code'].nunique()}")

    # Class capping — undersample majority classes to at most MAX_CLASS_COUNT examples.
    # Applied here so the dataset parquet files remain untouched.
    if effective_cap > 0:
        # sample(frac=1) shuffles the dataset first so head(n) draws a random
        # subset rather than always the first N rows of each class.
        # groupby().head(n) is used instead of groupby().apply(lambda g: g.sample(...))
        # because apply() can silently drop or rename columns in pandas 2.x.
        df = (
            df.sample(frac=1, random_state=RANDOM_STATE)
            .groupby("rome_code", sort=False)
            .head(effective_cap)
            .reset_index(drop=True)
        )
        print(f"✂️ After capping (max {effective_cap}/class): {len(df)} rows | {df['rome_code'].nunique()} classes")

    # X is the text content (features)
    X = df["text"].astype(str).values
    # y_raw is the original ROME code labels (targets) before encoding.
    y_raw = df["rome_code"].astype(str).values

    # Encode labels with LabelEncoder. This converts string labels to integers
    # (0..n_classes-1) to avoid colinearity issues and ensure compatibility with scikit-learn classifiers.
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"🏷️ Classes encoded: {n_classes}")

    X = np.asarray(X, dtype=object)
    y = np.asarray(y)

    # Split: train+val / test — stratify to maintain class distribution.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Split train+val into train / val.
    # Validation set is used for hyperparameter tuning; test set is held out for final evaluation only.
    val_ratio_on_trainval = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_on_trainval,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    print(f"✂️ Split sizes: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

    # ==============================
    # HYPERPARAMETER TUNING
    # ==============================
    print(f"\n{'='*60}")
    print(f"TUNING STRATEGY: {TUNING_STRATEGY.upper()}")
    print(f"{'='*60}\n")

    training_start_time = time.time()
    tuning_function = get_tuning_strategy(TUNING_STRATEGY)
    vectorizer, model, tuning_config = tuning_function(X_train, y_train, X_val, y_val)
    training_duration_seconds = time.time() - training_start_time

    # ==============================
    # FINAL EVALUATION
    # ==============================
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON ALL SPLITS")
    print(f"{'='*60}\n")

    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    def eval_split(name: str, Xv, yv) -> Dict[str, float]:
        y_pred = model.predict(Xv)
        acc = accuracy_score(yv, y_pred)
        f1m = f1_score(yv, y_pred, average="macro")
        scores = model.decision_function(Xv)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        top3 = top_k_accuracy_from_scores(scores, yv, k=min(3, scores.shape[1]))
        top5 = top_k_accuracy_from_scores(scores, yv, k=min(5, scores.shape[1]))
        print(f"📈 {name}: acc={acc:.4f} | f1_macro={f1m:.4f} | top3={top3:.4f} | top5={top5:.4f}")
        return {"accuracy": float(acc), "f1_macro": float(f1m), "top3": float(top3), "top5": float(top5)}

    metrics = {
        "train": eval_split("train", X_train_vec, y_train),
        "val":   eval_split("val",   X_val_vec,   y_val),
        "test":  eval_split("test",  X_test_vec,  y_test),
    }

    # ==============================
    # SAVE ARTIFACTS
    # ==============================
    print(f"\n💾 Saving artifacts to: {model_base_key}/")
    write_joblib(f"{model_base_key}/vectorizer.joblib", vectorizer)
    write_joblib(f"{model_base_key}/model.joblib", model)
    write_joblib(f"{model_base_key}/label_encoder.joblib", le)

    config = {
        "dataset_key": dataset_key,
        "dataset_dt": dt,
        "effective_version": effective_version,
        "capping": {"max_class_count": effective_cap if effective_cap > 0 else None},
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
        "backend": os.getenv("STORAGE_BACKEND", "local"),
        "rows": int(len(df)),
        "classes": int(n_classes),
        "tuning_strategy": TUNING_STRATEGY,
        "training_duration_seconds": float(training_duration_seconds),
        "training_duration_minutes": float(training_duration_seconds / 60),
    }

    write_json(f"{model_base_key}/metrics.json", metrics)
    write_json(f"{model_base_key}/config.json", config)
    write_json(f"{model_base_key}/train_meta.json", train_meta)

    if update_latest:
        write_json(
            f"{MODEL_PATH_PREFIX}/{MODEL_NAME}/LATEST.json",
            {"latest": effective_version, "updated_utc": run_ts},
        )
        logger.info("LATEST.json updated → %s", effective_version)
    else:
        logger.info("LATEST.json not updated (--no-update-latest). Version: %s", effective_version)

    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Version:       {effective_version}")
    print(f"Strategy:      {TUNING_STRATEGY}")
    print(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test F1-Macro: {metrics['test']['f1_macro']:.4f}")
    print(f"Artifacts:     {model_base_key}/")
    print(f"LATEST.json:   {'updated' if update_latest else 'NOT updated (experiment mode)'}")
    print(f"{'='*60}\n")

    return {
        "dt": dt,
        "dataset_key": dataset_key,
        "effective_version": effective_version,
        "rows": int(len(df)),
        "classes": int(n_classes),
        "metrics": metrics,
        "model_base_key": model_base_key,
        "tuning_strategy": TUNING_STRATEGY,
        "training_duration_seconds": float(training_duration_seconds),
        "latest_updated": update_latest,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train ROME classifier from Gold dataset."
    )
    parser.add_argument(
        "--dt",
        default=None,
        help="Dataset partition date (YYYY-MM-DD). Omit for auto (latest Gold dataset dt).",
    )
    parser.add_argument(
        "--max-class-count",
        type=int,
        default=None,
        help="Cap majority classes to at most N examples before training. Overrides MAX_CLASS_COUNT env var. 0 = no cap.",
    )
    parser.add_argument(
        "--no-update-latest",
        action="store_true",
        default=False,
        help="Do not update LATEST.json after training. Use for experiment runs (cap tuning, ablations).",
    )
    parser.add_argument(
        "--tuning-strategy",
        default=None,
        choices=["none", "manual", "grid", "random"],
        help="Tuning strategy. Omit to use TUNING_STRATEGY env var (default: none).",
    )
    args = parser.parse_args()
    if args.tuning_strategy is not None:
        os.environ["TUNING_STRATEGY"] = args.tuning_strategy
    result = train(args.dt, update_latest=not args.no_update_latest, max_class_count=args.max_class_count)
    logger.info("Training completed: %s", result)


if __name__ == "__main__":
    main()
