
# Job Market – Jan 26  
## ROME Code Prediction (France Travail)

This repository contains the **ML and ingestion module** of the Job Market project.
Its objective is to collect job offers from the France Travail API and train a machine learning model to predict the corresponding **ROME job code** from textual content.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py <- To create dataset for modeling
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py


---

# Project Overview

This module implements:

- Data ingestion from France Travail API
- Storage in a Bronze/Gold architecture
- Dataset preparation for ML
- TF-IDF + LinearSVC training
- Model versioning
- Local or S3/MinIO compatibility

---

# Data Ingestion Format – Why JSONL?

During ingestion, job offers retrieved from the France Travail API are stored in **JSONL (Newline Delimited JSON)** format.

Each line contains a single JSON object:

{"id": "048KLTP", "intitule": "Data Engineer", "romeCode": "M1805"}

### Why JSONL?

We chose JSONL for the Bronze layer because:

- Scalable – Each record is independent and written in immutable part files (`part-000001.jsonl`).
- Memory efficient – Files can be streamed line by line without loading the entire dataset into memory.
- S3-compatible – Object storage systems (MinIO/S3) do not support efficient file appends.
- Data lake friendly – Supports partitioning (`dt=...`, `run_id=...`).

---

# Data Pipeline

API → JSONL (Bronze) → Parquet (Gold) → ML Training

- Bronze layer: raw JSONL files
- Gold layer: cleaned dataset stored as Parquet for ML efficiency

---

# Machine Learning Model

## Problem Type

Multi-class text classification:

- ~307,000 job offers
- ~490 ROME classes
- High-dimensional sparse feature space

## Feature Engineering

We use **TF-IDF vectorization**:

- Unigrams and bigrams
- min_df filtering to remove rare noise
- max_features limit to control memory
- dtype=float32 to reduce RAM usage

## Model Choice – LinearSVC

We selected:

LinearSVC()

Why?

- Efficient for high-dimensional sparse data
- Scales well with large datasets
- Memory-efficient compared to Logistic Regression
- Strong performance for text classification

---

# Dataset Splitting

We use stratified splitting to preserve class distribution:

- 80% Training
- 10% Validation
- 10% Test

---

# Storage Backend

The project supports:

Local filesystem (development)

S3/MinIO object storage (production-ready)

Configured via environment variables.

---

# Running the Module

## 1. Ingest Data
python -m src.ingest.ingest_france_travail

## 2. Build Dataset
python -m src.data.make_dataset

## 3. Train Model
python -m src.models.train_model

---

# Model Versioning

Each training run saves artifacts in:

models/rome_tfidf/versions/vX/

A LATEST.json file indicates the active model version.

---


# Model Evaluation Metrics

To properly evaluate the ROME classification model, we use several complementary metrics.

---

## 1️⃣ Accuracy

Accuracy measures the proportion of correct predictions:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

It answers:

> How often is the model correct overall?

In our case (~490 classes), accuracy ≈ 78% means that 78% of job offers are assigned the correct ROME code.

However, accuracy alone can be misleading in multi-class or imbalanced datasets.

---

## 2️⃣ Precision

Precision measures how reliable the model’s predictions are for a given class.

Precision = TP / (TP + FP)

It answers:

> When the model predicts a given ROME code, how often is it correct?

High precision means few false positives.

---

## 3️⃣ Recall

Recall measures how well the model detects all true instances of a class.

Recall = TP / (TP + FN)

It answers:

> Among all real examples of a ROME code, how many did the model correctly identify?

High recall means few false negatives.

---

## 4️⃣ F1 Score

The F1 score balances precision and recall:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

It is high only when both precision and recall are high.

The F1 score is particularly useful when dealing with imbalanced classes.

---

## 5️⃣ Macro F1 (Multi-Class Case)

In a multi-class problem, we compute F1 for each class separately, then average:

F1_macro = (1/N) × Σ F1_i

Where:
- N = number of classes (~490)
- F1_i = F1 score for class i

Macro F1 gives equal importance to rare and frequent ROME codes.

This explains why Macro F1 (~67%) is lower than Accuracy (~78%):  
rare job codes are more difficult to predict.

---

# 🎯 Accuracy vs Precision – Archery Analogy

Imagine a target in archery:

- 🎯 The center represents the correct class.
- 🏹 The arrows represent the model’s predictions.

| Situation | Arrow Pattern | Accuracy | Precision | Interpretation |
|------------|--------------|-----------|------------|----------------|
| 🎯 Ideal Model | Arrows tightly grouped at the center | High | High | Correct and consistent predictions |
| 🎯 Off-Center Cluster | Arrows tightly grouped but slightly off center | Low | High | Consistent but systematically wrong |
| 🎯 Scattered Around Center | Arrows spread around the center | Moderate | Low | Sometimes correct but inconsistent |
| 🎯 Random | Arrows scattered everywhere | Low | Low | Unreliable model |

Key difference:

- Accuracy measures how close predictions are to the true class (hitting the center).
- Precision measures how tightly grouped predictions are for a given class.
- A model can be precise but not accurate (consistently wrong).
- A model can be accurate overall but imprecise for certain classes.

In multi-class classification, precision is computed per class, not geometrically, but the analogy helps illustrate the concept.

---

## 6️⃣ Top-K Accuracy

Top-K accuracy evaluates whether the correct ROME code appears among the top K predictions:

Top-K = (Number of samples where true class is in top K) / (Total number of samples)

In our project:

- Top-3 ≈ 89%
- Top-5 ≈ 92%

This means that in more than 90% of cases, the correct ROME code is among the 5 most likely predictions.

Top-K is especially relevant for:

- Assisted classification systems
- Human-in-the-loop workflows
- Recommendation interfaces

---

# 📊 Overall Interpretation

- Accuracy ≈ 78% → strong overall performance.
- Macro F1 ≈ 67% → rare classes remain more difficult.
- Top-5 ≈ 92% → excellent suitability for decision-support scenarios.
- Validation ≈ Test → good generalization and no major overfitting.

---