# Job Market – Architecture complète et guide de construction

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture générale](#architecture-générale)
3. [Stack technique](#stack-technique)
4. [Arborescence du projet](#arborescence-du-projet)
5. [Briques de fonctionnement](#briques-de-fonctionnement)
6. [Variables d'environnement](#variables-denvironnement)
7. [Formats de données](#formats-de-données)
8. [Choix techniques](#choix-techniques)
9. [Pipelines de données](#pipelines-de-données)
10. [Guide de déploiement](#guide-de-déploiement)

---

## Vue d'ensemble

**Job Market** est une plateforme de **d'analyse du marché du travail en France** construite autour de trois sources de données principales :

1. **France Travail** : API officielle française des offres d'emploi
2. **Welcome to the Jungle** : Scraping de plateforme d'emploi (sitemap + crawling)
3. **Référentiel ROME** : Nomenclature officielle des métiers français

### Objectifs principaux

- 🎯 **Ingérer** les offres d'emploi depuis plusieurs sources (APIs, web scraping)
- 🤖 **Prédire** le code ROME (nomenclature métiers) pour chaque offre
- 💾 **Stocker** les données dans une architecture multi-couches (bronze/silver/gold)
- 📊 **Monitorer** l'ingestion via dashboards Grafana temps réel
- 🔄 **Fusionner** les données de sources différentes
- 📊 **Restituer** des états d'indicateur maché 

---

## Architecture générale

```
┌──────────────────────────────────────────────────┐
│             CLIENTS / FRONTEND                   │
└─────────────────────┬────────────────────────────┘
                      │
        ┌─────────────┼───────────┐
        │             │           │
    ┌───▼──┐     ┌────▼────┐   ┌───▼──────┐
    │ API  │     │ Jupyter │   │ Grafana  │
    │:8000 │     │ :8888   │   │ :3000    │
    └───┬──┘     └────┬────┘   └───┬──────┘
        │             │           │
        └─────────────┼───────────┘
                      │
        ┌─────────────┼───┐
        │                 │
    ┌───▼────────┐   ┌────▼─────┐ 
    │ PostgreSQL │   │  MinIO   │ 
    │ :5432      │   │  :9000   │ 
    └────────────┘   └──────────┘ 
```

### Composants principaux

| Composant | Port | Rôle |
|----------|------|------|
| **FastAPI (jobmarket-api)** | 8000 | Endpoint d'ingestion, prédiction, monitoring |
| **PostgreSQL** | 5432 | Base de données principale (métrologie: job_runs, ingestion_logs) + Modèle en étoile de la couche Gold |
| **MinIO (S3-compatible)** | 9000 | Stockage datalake (données bronze/silver/gold) |
| **Grafana** | 3000 | Dashboards de monitoring en temps réel |
| **Jupyter Lab** | 8888 | Notebooks d'exploration et analyse |
| **pgAdmin** | 5050 | Admin PostgreSQL |

---

## Stack technique

### Backend & Ingestion

| Technologie | Version | Utilisation |
|-------------|---------|-------------|
| Python | 3.11 | Langage principal (stable, largement adpoté, supporté par les libs majeures|
| FastAPI | (latest) | Framework API REST (swagger automatique, typing natif) |
| Uvicorn | (latest) | ASGI server, choix naturel pour de l'async (projet I/O dépendant))  |
| pandas | ~2.x | Traitement de données |
| PyArrow | ~10.x | Format Parquet (silver/gold) |
| boto3 | ~1.x | Client S3/MinIO |
| psycopg | 3.2.6 | Driver PostgreSQL (v3, moderne, async natif) |
| psycopg-pool | 3.2.5 | Pool de connexions pour psycopg |
| requests | ~2.x | Requêtes HTTP |

### ML & Classification

| Technologie | Version | Utilisation |
|-------------|---------|-------------|
| scikit-learn | ~1.x | Classification TF-IDF + LinearSVC |
| joblib | ~1.x | Sérialisation modèles (standard recommandé pour scikit-learn) |
| transformers | ~4.x | NER (extraction skills, optionnel) |
| ollama | (latest) | LLM local pour extraction skills (optionnel)|

### Stockage & Monitoring

| Technologie | Version | Utilisation |
|-------------|---------|-------------|
| PostgreSQL | 15 | Base données structurées |
| MinIO | (latest) | Stockage S3-compatible |
| Grafana | 10.2.3 | Dashboards temps réel |
| Loki (deprecated) | - | Remplacé par PostgreSQL |

### Containerisation

| Technologie | Version |
|-------------|---------|
| Docker | (latest) |
| Docker Compose | ~3.9 |

---

## Arborescence du projet

```
jan26_bde_jobmarket/
│
├── .env                              # Configuration principale (variables d'env)
├── .gitignore                        # Git ignore patterns
├── docker-compose.yml                # Orchestration services (postgres, minio, grafana, api)
├── Dockerfile                        # Image Jupyter
├── Dockerfile.api                    # Image API FastAPI
├── requirements.txt                  # Dépendances Python
├── README.md                         # Documentation projet
│
├── minio-data/                       # Données brutes/traitées
│   ├── france_travail/               # FT: offres, codes ROME
│   ├── welcometothejungle/           # WTTJ: offres jobs, companies
│   
├── src/                              # Code source principal
│   ├── __init__.py
│   │
│   ├── api/                          # Endpoints REST (FastAPI)
│   │   ├── main.py                   # Routeurs: predict, ingest/*, data/*
│   │   ├── models.py                 # Pydantic models (PredictRequest, etc.)
│   │   └── README.md
│   │
│   ├── config/                       # Configuration projet
│   │   └── env.py                    # Chargement .env, project root detection
│   │
│   ├── ingest/                       # Pipelines ingestion données
│   │   ├── bronze/                   # Couche bronze (données brutes)
│   │   │   ├── france_travail.py     # Ingest FT offres (APIs + time windows)
│   │   │   ├── france_travail_rome_metiers.py  # Ingest codes ROME
│   │   │   ├── welcome_to_the_jungle.py        # Ingest WTTJ (sitemap + crawl)
│   │   │
│   │   ├── clients/                  # Clients API externes
│   │   │   └── france_travail_client.py   # OAuth2 + API FT
│   │   │
│   │   ├── data_models/              # Schémas données pour l'ingestion
│   │   │
│   │   ├── silver/                   # Couche silver (données transformées) . Production WTTJ au format silver + Ecriture en parquet
│   │   │
│   │   ├── tools/                    # Utilitaires (rate limiter, etc.)
│   │   │
│   │   ├── investigate/              # Scripts d'analyse/debug
│   │   │
│   │   └── tests/                    # Tests unitaires ingestion
│   │
│   ├── data/                         # Pipeline données (feature engineering)
│   │   ├── make_dataset.py           # Création datasets ML
│   │   └── make_merge_dataset_ft_wttj_with_rome.py  # Fusion FT + WTTJ (silver parquet) => silver parquet
│   │
│   ├── features/                     # Feature engineering
│   │   └── build_features.py
│   │
│   ├── models/                       # ML pipeline
│   │   ├── predict_model.py          # Inférence TF-IDF + LinearSVC
│   │   └── train_model.py            # Entraînement modèles
│   │
│   ├── utils/                        # Utilitaires globaux
│   │   ├── log_to_db.py              # Logging PostgreSQL centralisé
│   │   ├── text_processing.py        # Nettoyage texte
│   │   └── data_prefix_resolver.py   # Résolution chemins S3
│   │
│   ├── storage/                      # Abstraction stockage
│   │   └── storage.py                # Interface Storage (local/S3)
│   │
│   ├── visualization/                # Visualisations
│   │   └── visualize.py
│   │
│   └── observability/                # Monitoring
│       └── job_store.py              # Tracking job status
│
├── models/                           # Modèles ML sérialisés
│   ├── rome_tfidf_v2/                # TF-IDF + LinearSVC model
│   └── artifacts/                    # Artifacts du modèle de ML
│
├── notebooks/                        # Analyses Jupyter
│   ├── 00_exploration/
│   ├── 01_debug/
│   └── 02_reports/
│
├── postgres/                         # Scripts init PostgreSQL
│   ├── init/
│       ├── 001_job_runs.sql          # Schéma job_runs
│       └── 002_ingestion_logs.sql    # Schéma ingestion_logs
│
├── grafana/                          # Configuration Grafana
│   ├── provisioning/                 # Provisioning automatique
│   │   ├── datasources/
│   │   │   └── postgres.yml          # Datasource JobStore PostgreSQL
│   │   └── dashboards/
│   │       └── providers.yml         # Configuration dashboards
│   │
│   └── dashboards/                   # Dashboard JSON files
│       ├── rome-ingestion-logs.json      # Dashboard ingestion ROME
│       ├── france-travail-offers-logs.json   # Dashboard FT offres
│       ├── wttj-ingestion-logs.json        # Dashboard WTTJ offres
│       └── import-jobs.json                # Dashboard jobs merged
│
├── pgadmin/                          # Configuration pgAdmin
│   └── servers.json                  # Connexion PostgreSQL
│
├── logs/                             # Logs applicatifs
│   ├── api/
│   ├── ingestion/
│   └── prediction/
│
├── references/                       # Documentation externe
│
├── reports/                          # Rapports d'analyse
│   └── figures/
│
├── exploration/                      # Scripts d'exploration
│
├── ollama-data/                      # Modèles Ollama (skills extraction)
│
└── LICENSE
```

### Structure des données brutes (S3/MinIO)

```
jobmarket/
├── france_travail/
│   ├── bronze/
│   │   ├── offers/
│   │   │   └── dt=YYYY-MM-DD/
│   │   │       └── run_id=YYYYMMDDTHHMMSSz/
│   │   │           ├── code_rome=C1504/
│   │   │           │   └── segment=global/
│   │   │           │       └── part-000001.jsonl
│   │   │           └── ...
│   │   └── metadata/
│   │       ├── rome_metiers.jsonl    # Codes ROME (532 codes)
│   │       └── ...
│   │
│   └── silver/                       # (Couche transformée - Parquet)
│       └── dt=YYYY-MM-DD/
│
├── welcometothejungle/
│   ├── bronze/
│   │   ├── jobs/
│   │   │   └── dt=YYYY-MM-DD/
│   │   ├── companies/
│   │   └── ...
│   └── silver/
│
└── gold/
    ├── datasets/
    │   ├── rome_dataset.parquet      # Dataset ROME pour ML
    │   └── ft_wttj_merged.parquet    # Dataset fusionné FT+WTTJ
    └── ...
```

---

## Briques de fonctionnement

### 1. Ingestion France Travail (`france_travail.py`)

**Objectif**: Importer les offres d'emploi via l'API officielle France Travail

**Processus** :
1. OAuth2 vers France Travail (API_KEY + API_SECRET)
2. Récupération codes ROME disponibles
3. Pour chaque code ROME :
   - Requête API avec time windows (7 jours par défaut)
   - Si > 3150 résultats : split binaire récursif
   - Construction JSONL par segment
4. Storage : `france_travail/bronze/offers/dt=YYYY-MM-DD/run_id=*/code_rome=*/*.jsonl`

**Configuration** :
- `FT_RATE_LIMIT_RPS`: 10 requêtes/seconde
- `FT_WINDOW_DAYS`: 7 jours par fenêtre
- `FT_MAX_WINDOWS`: 260 fenêtres (5 ans)
- `FT_MAX_ROME_CODES`: 0 = tous les codes

**Logging** :
```python
log_to_db(
    endpoint='france_travail_offers',
    level='INFO'/'ERROR',
    message="✅ X offres importées (Y codes ROME, Z appels, W erreurs)",
    task_id=task_id,
    duration_sec=...,
    records_count=...,
    error_count=...
)
```

---

### 2. Ingestion Welcome to the Jungle (`welcome_to_the_jungle.py`)

**Objectif**: Scraper les offres WTTJ via sitemap et crawling

**Processus** :
1. Télécharger sitemap index XML
2. Filtrer sitemaps (jobs_en, jobs_fr, companies_*)
3. Extraire URLs depuis sitemaps
4. Crawler en parallèle (10 workers par défaut) :
   - Respecter rate limit (2 req/sec par défaut)
   - Extraire `window.__INITIAL_DATA__` (React embedded JSON)
   - Retry exponential backoff
5. Storage : `welcometothejungle/bronze/{jobs,companies}/dt=YYYY-MM-DD/part-*.jsonl`

**Configuration** :
- `WTTJ_RUN_MODE`: new | resume | incremental
- `WTTJ_WORKERS`: 10 threads parallèles
- `WTTJ_PART_SIZE`: 10 records/flush
- `WTTJ_RPS`: 2 requêtes/seconde
- `WTTJ_BURST`: 4 tokens burst
- `WTTJ_MAX_JOBS`: 0 = pas de limite
- `WTTJ_MAX_COMPANIES`: 0 = pas de limite
- `WTTJ_STORE_HTML`: never | on_error | always

**Logging** :
```python
log_to_db(
    endpoint='welcome_to_the_jungle',
    level='INFO'/'ERROR',
    message="✅ X offres importées (Y URLs, Z erreurs)",
    task_id=task_id,
    duration_sec=...,
    urls_processed=...
)
```

---

### 3. Ingestion Codes ROME (`france_travail_rome_metiers.py`)

**Objectif**: Importer la nomenclature ROME (532 codes métiers)

**Processus** :
1. Appel API France Travail : `/partenaire/rome-metiers/v1/metiers/metier`
2. Paramètres : `champs=code,libelle`
3. Deduplique les codes (clé primaire = code)
4. Storage : `france_travail/bronze/metadata/rome_metiers.jsonl`

**Logging** :
```python
log_to_db(
    endpoint='rome_metiers',
    level='INFO'/'ERROR',
    message="✅ X codes ROME métiers importés",
    task_id=task_id,
    records_count=X
)
```

---

### 4. Prédiction ROME (`/predict` endpoint)

**Objectif**: Classifier automatiquement une offre vers son code ROME

**Modèle ML**:
- **Algorithme**: TF-IDF + LinearSVC
- **Features**: Intitulé + Description + Compétences
- **Top-K**: 5 codes ROME les plus probables

**Processus** :
1. Build payload texte (intitulé + description + compétences)
2. Vectorisation TF-IDF pré-entraînée
3. Prédiction LinearSVC
4. Retour top-5 codes avec scores

**Hyperparamètres TF-IDF**:
- `TFIDF_NGRAM_MIN`: 1
- `TFIDF_NGRAM_MAX`: 2
- `TFIDF_MIN_DF`: 5
- `TFIDF_MAX_DF`: 0.9
- `TFIDF_MAX_FEATURES`: 200000

---

### 5. Fusion Datasets (`make_merge_dataset_ft_wttj_with_rome.py`)

**Objectif**: Fusionner les données FT + WTTJ et associer codes ROME

**Processus** :
1. Charger bronze FT (JSONL compressé)
2. Charger silver WTTJ (Parquet)
3. Harmoniser schémas
4. Matcher avec codes ROME via prédiction
5. Stocker résultat : `gold/datasets/ft_wttj_merged.parquet`

**Configuration** :
- `FT_BRONZE_PREFIX`: chemin auto ou manuel
- `WTTJ_SILVER_PREFIX`: chemin auto ou manuel
- `MERGED_DATASET_S3_PREFIX`: gold
- `FT_READ_WORKERS`: 20 workers (parallélisme)
- `WTTJ_READ_WORKERS`: 20 workers

---

### 6. Monitoring Ingestion (`ingestion_logs` table)

**Objectif**: Logger centralisé des opérations ingestion dans PostgreSQL

**Table Structure**:
```sql
CREATE TABLE ingestion_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    endpoint VARCHAR(50),           -- 'rome_metiers', 'france_travail_offers', 'welcome_to_the_jungle'
    level VARCHAR(10),               -- 'INFO', 'WARNING', 'ERROR'
    task_id VARCHAR(100),            -- ID unique de tâche
    message TEXT,                    -- Message log
    duration_sec NUMERIC(10, 3),     -- Durée exécution
    records_count BIGINT,            -- Nombre records traités
    error_count INTEGER,             -- Nombre erreurs
    extra_metadata JSONB             -- Métadonnées spécifiques
);
```

**Utilisation** :
```python
from src.utils.log_to_db import log_to_db

log_to_db(
    endpoint='france_travail_offers',
    level='INFO',
    message="✅ 100 offres importées",
    task_id="task-123",
    duration_sec=45.2,
    records_count=100,
    error_count=0,
    rome_processed=50,      # metadata extra
    api_calls=52
)
```

**Indexes**:
- `idx_ingestion_logs_timestamp` : tri chronologique Grafana
- `idx_ingestion_logs_endpoint` : filtrer par source
- `idx_ingestion_logs_level` : filtrer par sévérité
- `idx_ingestion_logs_task_id` : tracker tâche

**Retention**: 30 jours (fonction `cleanup_old_ingestion_logs()`)

**Dashboards Grafana** :
- `rome-ingestion-logs.json` : Logs ingestion ROME
- `france-travail-offers-logs.json` : Logs offres FT
- `wttj-ingestion-logs.json` : Logs offres WTTJ

**Docker reconstruction du volume**:
Supprime la base et relance les scripts d'initialisation (/postgres/init)
```python
docker compose down

# Syntaxe volume name : nomProjet_nomVolume
# Suppression du volume de persistance de données
docker volume rm jan26_bde_jobmarket_jobdb-data

docker compose up -d postgres
docker logs jobmarket-postgres
```

---

### 7. Job Store (`job_runs` table)

**Objectif**: Tracker les exécutions de jobs (ingestion, ML, etc.)

**Table Structure**:
```sql
CREATE TABLE job_runs (
    run_id TEXT PRIMARY KEY,
    job_type TEXT,                  -- 'import', 'transform', 'predict'
    source TEXT,                    -- 'rome_metiers', 'france_travail_offers', etc.
    status TEXT,                    -- 'RUNNING', 'SUCCESS', 'FAILED'
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    duration_ms BIGINT,
    progress_pct INT,
    message TEXT,
    records_count BIGINT DEFAULT 0,
    pages_count BIGINT DEFAULT 0,
    errors_count BIGINT DEFAULT 0,
    records_per_sec NUMERIC(10, 3),
    params_json JSONB,              -- Paramètres d'exécution
    result_json JSONB,              -- Résultat final
    error_text TEXT,
    updated_at TIMESTAMPTZ
);
```

**Indexes**:
- `idx_job_runs_started_at` : requêtes temporelles
- `idx_job_runs_status` : filtrer par statut
- `idx_job_runs_source` : filtrer par source
- `idx_job_runs_records_per_sec` : analyse perf

---

### 8. Storage Abstraction (`storage.py`)

**Interface Storage**:
- `write_bytes(key, payload)` : Écrire fichier brut
- `read_bytes(key)` : Lire fichier brut
- `write_jsonl(key, records)` : Écrire JSONL
- `read_jsonl(key)` : Lire JSONL
- `write_parquet(key, dataframe)` : Écrire Parquet
- `read_parquet(key)` : Lire Parquet
- `list_keys(prefix)` : Lister clés

**Implémentations**:
1. **LocalStorage** : Système fichiers (`FT_DATA_DIR`)
2. **S3Storage** : MinIO/S3 (boto3)

**Configuration** :
- `STORAGE_BACKEND`: "local" | "S3"
- Si S3 :
  - `S3_ENDPOINT_URL`: http://minio:9000 (Docker)
  - `S3_BUCKET`: "jobmarket"
  - `S3_PREFIX_FT`: "france_travail"
  - `S3_PREFIX_WTTJ`: "welcometothejungle"
  - `S3_MAX_POOL_CONNECTIONS`: 50 workers

---

## Variables d'environnement

### Section 1: France Travail OAuth & API

```env
# OAuth2 France Travail
API_KEY="XXXXXXXXXXX"
API_SECRET="XXXXXXXXXXXXX"
FT_SCOPE="api_offresdemploiv2 o2dsoffre nomenclatureRome api_rome-metiersv1"
FT_TOKEN_URL="https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
FT_API_BASE_URL="https://api.francetravail.io"

# Rate limiting (requêtes/sec)
FT_RATE_LIMIT_RPS=10

# Ingestion parameters
FT_WINDOW_DAYS=7                    # Fenétres temps (jours)
FT_MAX_WINDOWS=260                  # Max fenêtres (5 ans)
FT_BINARY_SPLIT_MIN_SECONDS=3600    # Split interval si > 3150 results
FT_MAX_ROME_CODES=0                 # 0 = tous les codes

# Data storage
FT_DATA_DIR=data/france_travail     # Chemin local
```

### Section 2: Welcome to the Jungle

```env
# Stockage local & S3
WTTJ_DATA_DIR=data/welcometothejungle
S3_PREFIX_WTTJ=welcometothejungle

# Mode exécution
WTTJ_RUN_MODE=new                   # new | resume | incremental
WTTJ_RUN_ID=                        # Requis si resume
WTTJ_RESUME_FROM_RUN_ID=            # Requis si incremental

# Crawl settings
WTTJ_WORKERS=10                     # Threads parallèles
WTTJ_PART_SIZE=10                   # Records/flush
WTTJ_RPS=2                          # Requêtes/sec
WTTJ_BURST=4                        # Burst tokens

# Retry HTTP
WTTJ_RETRIES=5
WTTJ_BACKOFF=0.6

# HTML storage
WTTJ_STORE_HTML=always              # never | on_error | always

# Debug
WTTJ_MAX_JOBS=15                    # 0 = illimité
WTTJ_MAX_COMPANIES=0                # 0 = illimité
```

### Section 3: Storage Backend

```env
STORAGE_BACKEND=S3                  # local | S3

# MinIO/S3 (utilisé si STORAGE_BACKEND=S3)
S3_ENDPOINT_URL=http://localhost:9000           # Local MinIO
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin123
S3_BUCKET=jobmarket
S3_PREFIX_FT=france_travail
S3_PREFIX_WTTJ=welcometothejungle
S3_REGION=us-east-1
S3_MAX_POOL_CONNECTIONS=50          # Connection pool size
```

### Section 4: Dataset Merge & ML

```env
# Auto-détection chemin (vide) ou manuel
FT_BRONZE_PREFIX=                   # Auto ou 'bronze/offers/dt=2026-02-23'
WTTJ_SILVER_PREFIX=                 # Auto ou 'silver/dt=2026-02-21'

# Sortie merge
MERGED_DATASET_S3_PREFIX=silver
MERGED_DATASET_PREFIX=datasets/ft_wttj_merged

# Parallelisme lecture données
FT_READ_WORKERS=20
WTTJ_READ_WORKERS=20

# Données d'entraînement ML
DATASET_KEY=gold/datasets/rome_dataset.parquet
MODEL_NAME=rome_tfidf
MODEL_VERSION=v2
MIN_CLASS_COUNT=25
MAX_COMPETENCES=25

# API ML
ML_HOST_API=http://localhost:8000
ML_ENDPOINT=predict

# TF-IDF hyperparameters
TFIDF_NGRAM_MIN=1
TFIDF_NGRAM_MAX=2
TFIDF_MIN_DF=5
TFIDF_MAX_DF=0.9
TFIDF_MAX_FEATURES=200000

# TOP-K predictions
TOP_K=5
```

### Section 5: Skills Extraction (optionnel)

```env
ENABLE_SKILLS_EXTRACTION=true       # Extraction automatique compétences

# NER (Named Entity Recognition)
NER_MODEL_NAME=Jean-Baptiste/camembert-ner

# LLM (générateur compétences)
LITELLM_MODEL=ollama/phi3:mini
LLM_TIMEOUT=300
STORE_SKILLS_FEEDBACK=true
NER_DEVICE=auto
```

### Section 6: Logging & Monitoring

```env
LOG_LEVEL=INFO                      # DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_MAX_BYTES=10485760              # 10MB max par fichier log
LOG_BACKUP_COUNT=5                  # Nb fichiers backup avant rotation

# Grafana logs (structured JSON)
ENABLE_GRAFANA_LOGS=true            # Émettre logs JSON pour Loki/Grafana
```

### Section 7: Grafana Configuration

```env
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin
GF_USERS_ALLOW_SIGN_UP=false        # Pas d'auto-registration
GF_DATE_FORMATS_DEFAULT_TIMEZONE=browser   # Fuseau client (pas UTC)
```

### Section 8: PostgreSQL Configuration

```env
POSTGRES_DB=jobdb
POSTGRES_USER=jobuser
POSTGRES_PASSWORD=jobpass           # ⚠️ À changer en production !
JOBSTORE_DSN=postgresql://jobuser:jobpass@localhost:5432/jobdb
```

### Section 9: pgAdmin Configuration (optionnel)

```env
PGADMIN_DEFAULT_EMAIL=willramarques@gmail.com
PGADMIN_DEFAULT_USERNAME=jobuser
PGADMIN_DEFAULT_PASSWORD=jobpass    # ⚠️ À changer en production !
```

---

## Formats de données

### 1. Format JSONL (Ligne-délimité JSON)

**Utilisation**: Bronze layer (données brutes)

**Structure générale**:
```json
{
  "id": "unique-identifier",
  "title": "...",
  "description": "...",
  "url": "...",
  "source_metadata": { ... },
  "ingestion_timestamp": "2026-02-26T17:36:00Z",
  "run_id": "YYYYMMDDTHHMMSSz"
}
```

**Champs communs tous sources**:
- `id` : Identifiant unique (URL hash, API ID, etc.)
- `title` / `intitule` : Libellé du poste
- `description` : Description détaillée
- `location` : Localisation
- `salary` : Rémunération
- `contract_type` : Type contrat
- `source` : Source données (france_travail, wttj, etc.)
- `ingestion_timestamp` : Date/heure ingestion UTC

**France Travail** :
```json
{
  "id": "108123456",
  "title": "Data Scientist",
  "description": "Nous recherchons un Data Scientist...",
  "rome_code": "M1602",
  "rome_label": "Analyse de données",
  "location": "Île-de-France",
  "employer": "TechCorp",
  "contract_type": "CDI",
  "salary_min": 45000,
  "salary_max": 55000,
  "publication_date": "2026-02-25",
  "source": "france_travail",
  "run_id": "20260226T173600Z"
}
```

**Welcome to the Jungle** :
```json
{
  "id": "www.welcometothejungle.com/jobs/...",
  "title": "Senior Backend Engineer",
  "description": "...",
  "company": "StartupXYZ",
  "location": "Paris",
  "contract_type": "CDI",
  "salary": "50k-70k€/an",
  "url": "https://www.welcometothejungle.com/...",
  "posted_at": "2026-02-20",
  "source": "wttj",
  "run_id": "20260226T173600Z"
}
```

### 2. Format Parquet (Silver/Gold layers)

**Utilisation**: Données transformées et alignées (pandas DataFrames)

**Compression**: Snappy (défaut)

**Schéma exemple**:
```python
import pyarrow as pa

schema = pa.schema([
    ('id', pa.string()),
    ('title', pa.string()),
    ('description', pa.string()),
    ('rome_code', pa.string()),          # Prédiction du modèle
    ('rome_label', pa.string()),
    ('rome_confidence', pa.float32()),   # Score prédiction (0-1)
    ('location', pa.string()),
    ('salary_min', pa.int64()),
    ('salary_max', pa.int64()),
    ('source', pa.string()),
    ('url', pa.string()),
    ('ingestion_date', pa.timestamp('ns')),
    ('run_id', pa.string()),
    ('metadata', pa.struct([         # Metadata flexible (JSON-like)
        ('skills', pa.list_(pa.string())),
        ('seniority', pa.string()),
        ('experience_years', pa.int32()),
        ('industry', pa.string()),
    ]))
])
```

### 3. Format PostgreSQL (Schéma job_runs)

```sql
-- Exemple enregistrement
INSERT INTO job_runs (
    run_id,
    job_type,
    source,
    status,
    started_at,
    ended_at,
    duration_ms,
    progress_pct,
    message,
    records_count,
    pages_count,
    errors_count,
    records_per_sec,
    params_json,
    result_json
) VALUES (
    'france-travail-20260226T173600Z',
    'import',
    'france_travail_offers',
    'SUCCESS',
    '2026-02-26 17:36:00+01',
    '2026-02-26 18:15:30+01',
    2370000,
    100,
    '✅ 15000 offres importées (50 codes ROME)',
    15000,
    52,
    2,
    6.33,
    '{"window_days": 7, "max_rome_codes": 0}',
    '{"written": 15000, "calls": 52, "errors": 2}'
);
```

### 4. Format Logs Structurés (JSON pour Grafana)

```json
{
  "timestamp": "2026-02-26T17:36:00Z",
  "event_type": "job_finished",
  "run_id": "rome-metiers-20260226T173600Z",
  "source": "rome_metiers",
  "status": "SUCCESS",
  "progress_pct": 100,
  "records_count": 532,
  "pages_count": 0,
  "errors_count": 0,
  "duration_sec": 15.2
}
```

---

## Choix techniques

### 1. **PostgreSQL vs NoSQL**

✅ **PostgreSQL choisi** car :
XXX

### 2. **S3/MinIO vs Base de données directe**

✅ **S3 pour données brutes** car :
- Données volumineuses (millions offres) → coûteux en DB
- Archivage à long terme
- Partitionnement naturel (dt=YYYY-MM-DD)
- Scalabilité horizontale
- Support formats multiples (JSONL, Parquet)

✅ **PostgreSQL pour monitoring** car :
- Queries analytiques Grafana
- Tracking job status (job_runs, ingestion_logs)
- Logs centralizés avec timestamps

### 3. **JSONL vs Parquet**

| Format | Bronze | Silver | Gold |
|--------|--------|--------|------|
| **JSONL** | ✅ | ❌ | ❌ |
| **Parquet** | ❌ | ✅ | ✅ |

Raisons :
- **JSONL (Bronze)** : Flexible, incrémental append-only, direct from APIs
- **Parquet (Silver/Gold)** : Compression 5-10x, colonnaire (analytics), compatible Spark

### 4. **FastAPI**

✅ **FastAPI** car :
- Type hints (Pydantic models)
- Auto-documentation (Swagger/OpenAPI)
- Async/await support
- Meilleure performance
- Modèle moderne pour APIs

### 5. **TF-IDF + LinearSVC vs Transformers/BERT**

✅ **TF-IDF + LinearSVC** car :
- **Perf** : Inférence < 1ms vs 500ms pour un transformer BERT
- **Simplicité** : CPU-only, pas GPU requis
- **Taille** : ~10MB vs 400MB pour BERT
- **Suffit** : Tâche de classification linéaire
- **Production** : Stable, maintenance plus facile

### 6. **PostgreSQL Logs vs Loki**

**Migration faite** : Loki → PostgreSQL car :
- ✅ Réduction de la complexité d'infrastructure (pas service Loki/Promtail)
- ✅ Requêtes SQL directes dans Grafana
- ✅ Stockage structuré avec indexes
- ✅ Rétention configurable (30j par défaut)
- ❌ Loki : Overkill pour notre volumétrie

### 7. **Docker Compose**

✅ **Docker Compose** car :
- Environnement dev/test simple
- Suffisant pour le projet
- Migration future possible

---

## Pipelines de données

### Pipeline 1 : Ingestion France Travail

```
┌─────────────────────────────────────────────────────────────┐
│  POST /ingest/france-travail-offers (avec task_id)          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  OAuth2 Token  │ ◄─── API_KEY + API_SECRET
        │  (1h validity) │
        └────────┬───────┘
                 │
                 ▼
    ┌─────────────────────────────┐
    │ Get ROME codes              │  → 532 codes
    │ (deduplicated)              │
    └────────────┬────────────────┘
                 │
                 ▼
    ┌───────────────────────────────────────┐
    │ For each ROME code:                   │
    │ - Build time windows (7d intervals)   │
    │ - Query API : /offres/search          │
    │   + rate limit: 10 req/sec            │
    │ - If result > 3150: Binary split      │
    │ - Write segments to JSONL             │
    │ - Update progress_callback            │
    └────────────┬──────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│ Storage: S3/MinIO                        │
│ Path: france_travail/bronze/offers/      │
│       dt=2026-02-26/                     │
│       run_id=20260226T173600Z/           │
│       code_rome=C1504/                   │
│       segment=global/                    │
│       part-000001.jsonl                  │
└───────┬──────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ PostgreSQL - job_runs table                 │
│ - status: SUCCESS/FAILED                    │
│ - records_count: 15000                      │
│ - duration_ms: 2370000                      │
│ - errors_count: 2                           │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ PostgreSQL - ingestion_logs table           │
│ - endpoint: france_travail_offers           │
│ - level: INFO/ERROR                         │
│ - message: "✅ 15000 offres importées"      │
│ - duration_sec: 2370                        │
│ - records_count: 15000                      │
│ - error_count: 2                            │
└─────────────────────────────────────────────┘
```

### Pipeline 2 : Ingestion Welcome to the Jungle

```
┌───────────────────────────────────────────────────────────────┐
│  POST /ingest/welcome-to-jungle (mode=new|resume|incremental) |
└────────────────┬──────────────────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │ Download sitemap_index   │
    │ https://www.w2j.com/     │
    │ sitemap_index.xml        │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────┐
    │ Filter sitemaps:                     │
    │ - jobs_en, jobs_fr                   │
    │ - companies_*                        │
    └────────────┬─────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ Extraire URLs from filtered maps   │
    │ (e.g. 100k URLs de jobs)           │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────┐
    │ Parallel crawl (10 workers):             │
    │ - Rate limit: 2 req/sec (WTTJ_RPS)       │
    │ - Retry: exponential backoff (5 try)     │
    │ - Extract window.__INITIAL_DATA__        │
    │ - HTML storage: always/on_error/never    │
    │ - Flush JSONL every 10 records           │
    └────────────┬─────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ Storage: S3/MinIO              │
    │ Path: welcometothejungle/      │
    │       bronze/jobs/             │
    │       dt=2026-02-26/           │
    │       part-001.jsonl           │
    └────────────┬───────────────────┘
                 │
                 ▼
┌…(similarly job_runs + ingestion_logs)…┐
```

### Pipeline 3 : Fusion Datasets (FT + WTTJ)

```
┌──────────────────────────────────────┐
│ POST /data/merge-datasets            │
└────────────┬─────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ Load Bronze FT (JSONL)          │
    │ Auto-detect or manual prefix    │
    │ FT_BRONZE_PREFIX                │
    │ → pandas DataFrame (X records)  │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────┐
    │ Load Silver WTTJ (Parquet)       │
    │ Auto-detect or manual prefix     │
    │ WTTJ_SILVER_PREFIX               │
    │ → pandas DataFrame (Y records)   │
    └────────────┬─────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │ Harmonize schemas:              │
    │ - Normalize column names        │
    │ - Map salary ranges             │
    │ - Location standardization      │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌───────────────────────────────────┐
    │ Concatenate & deduplicate         │
    │ Combined: X + Y records           │
    └────────────┬──────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────┐
    │ Predict ROME codes (via /predict API)   │
    │ For each record:                        │
    │ - Build payload (title+desc)            │
    │ - POST /predict                         │
    │ - Extract rome_code + confidence        │
    └────────────┬────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ Storage: S3/MinIO (Parquet)          │
│ Path: gold/datasets/                 │
│       ft_wttj_merged.parquet         │
│                                      │
│ Columns:                             │
│ - id, title, description             │
│ - source (ft|wttj)                   │
│ - rome_code, rome_label              │
│ - rome_confidence                    │
│ - location, salary, url              │
│ - ingestion_date, run_id             │
│ - skills, seniority, etc (metadata)  │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ PostgreSQL tracking (job_runs)       │
│ - source: "merge_datasets"           │
│ - records_count: X+Y                 │
│ - status: SUCCESS/FAILED             │
└──────────────────────────────────────┘
```

### Pipeline 4 : Prédiction ROME

```
┌──────────────────────────────────────┐
│ POST /predict                        │
│ {                                    │
│   "intitule": "Data Scientist",      │
│   "description": "...",              │
│   "competences": [...]               │
│ }                                    │
└────────────┬─────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ Build text payload:             │
    │ - Concatenate intitule, desc    │
    │ - Append competences            │
    │ - Clean & normalize             │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────┐
    │ TF-IDF Vectorization             │
    │ - Load pre-fit TF-IDF model      │
    │ - Transform text → sparse matrix │
    │ - 200k features max              │
    └────────────┬─────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────┐
    │ LinearSVC Prediction             │
    │ - sparse matrix → scores         │
    │ - Get top-5 classes              │
    │ - Normalize scores to proba      │
    └────────────┬─────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────┐
│ Response JSON                         │
│ {                                     │
│   "code_rome": "M1602",               │
│   "label": "Analyse de données",      │
│   "confidence": 0.87,                 │
│   "top_k": [                          │
│     {"code": "M1602", "score": 0.87}, │
│     {"code": "M1603", "score": 0.78}, │
│     ...                               │
│   ]                                   │
│ }                                     │
└───────────────────────────────────────┘
```

---

## Guide de déploiement

### Prérequis

- Docker + Docker Compose
- Python 3.11
- 4GB RAM minimum, 8GB recommandé
- 50GB disque (données + modèles ML)

### Installation locale

```bash
# 1. Clone repository
git clone https://github.com/...  /jan26_bde_jobmarket
cd jan26_bde_jobmarket

# 2. Variables d'environement
cp .env.example .env              # Adapter API_KEY, API_SECRET, etc

# 3. Environement virutel
cp .env.example .env              # Adapter API_KEY, API_SECRET, etc

# Activation
python -m venv env_job_market
source env_job_market/bin/activate  # Unix
# ou
env_job_market\Scripts\Activate.ps1  # Windows

# 3. Instalation des dépendances
pip install -r requirements.txt

# 4. Lancement des servcices Docker 
docker-compose up -d               # Postgres, MinIO, Grafana, API
En cas de modifications des variables dans le env il faut recréer le container:

docker-compose up -d --force-recreate jobmarket-api

# 5. Verifier le bon lancement des services
docker-compose ps                 
curl http://localhost:8000/docs   # FastAPI Swagger
curl http://localhost:3000/ # Dashboard Grafana
curl http://localhost:8888/ # Jupyter (token=jobmarket)

```

### Accès aux services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API FastAPI** | http://localhost:8000 | - |
| **Swagger UI** | http://localhost:8000/docs | - |
| **Grafana** | http://localhost:3000 | admin / admin |
| **pgAdmin** | http://localhost:5050 | jobuser / jobpass |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |
| **Jupyer Notebook** | http://localhost:8888 | token : jobmarket |

### Premiers tests

```bash
# 1. Ingestion ROME codes (async)
curl -X POST "http://localhost:8000/ingest/rome-metiers?background=true"

# 2. Checker status
curl "http://localhost:8000/ingest/status"

# 3. Prédiction ROME
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "intitule": "Data Scientist",
    "description": "Analysez les données avec Python et Machine Learning",
    "competences": ["Python", "SQL", "Scikit-learn"]
  }'

# 4. Ingestion offers async
curl -X POST "http://localhost:8000/ingest/france-travail-offers?background=true&max_rome_codes=5"

# 5. Monitoring dans Grafana
# - Aller  à http://localhost:3000
# - Puis dans dashboard
```


### Production Deployment

#### 1. Sécurité

```bash
# Changer credentials par défaut
GF_SECURITY_ADMIN_PASSWORD=<strong_password>
PGADMIN_DEFAULT_PASSWORD=<strong_password>
S3_SECRET_KEY=<random_key>
POSTGRES_PASSWORD=<strong_password>
```
---

## Résumé Architecture

```ascii
┌─────────────────────────────────────────────────────────────────┐
│                   CLIENTS / USERS                               │
│   - API users (prédiction)                                      │
│   - Data engineers (Grafana)                                    │
│   - Data engineers (Jupyter)                                    │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
      ┌──────▼──────┐                   ┌────────▼─────────┐
      │   API REST  │                   │    Monitoring    │
      │  FastAPI    │                   │   - Grafana      │
      │  :8000      │                   │   - Dashboards   │
      └──────┬──────┘                   └────────┬─────────┘
             │                                   │
             ▼                                   ▼
   ┌─────────────────────┐         ┌──────────────────────┐
   │   INGESTION LAYER   │         │   STORAGE LAYER      │
   │ ─────────────────── │         │ ──────────────────── │
   │ • France Travail    │────┐    │ • PostgreSQL         │
   │ • WTTJ              │    │    │   - job_runs         │
   │ • ROME Codes        │    │    │   - ingestion_logs   │
   │ • Prediction ROME   │    │    │                      │
   │ • Merge Datasets    │    │    │ • MinIO (S3)         │
   │ • ML Training       │    │    │   - Bronze (JSONL)   │
   └─────────────────────┘    │    │   - Silver (Parq.)   │
                              │    │   - Gold (Datasets)  │
                              └────│                      │
                                   └──────────────────────┘
```

---

## Checklist de build

- [ ] Cloner le repository
- [ ] Configurer`.env` (API_KEY, API_SECRET)
- [ ] `docker-compose up -d`
- [ ] Tester endpoint API `/docs`
- [ ] Suivre monitoring dans Grafana
- [ ] Lancer test ingestion (ROME codes)
- [ ] Vérifier logs

### dans Grafana
- [ ] Tester prédiction `/predict`
- [ ] Éxecuter ingestion FT/WTTJ (petit test)
- [ ] Fusion datasets
- [ ] Production: changer passwords, configurer backup

### dans PgAdmin
Pour vérifier si la connexion à la base postgres est OK
En cas de non connexion : cela apparait dans les logs

- [ ] Aller sur pgadmin pour créer une connection via "Query Tool Workspace"
- [ ] Créer une connexion au serveur : 
        - Host name/address = postgres
        - Port = 5432
        - Maintenance database = jobdb
        - Username = jobuser (cf infos de connexion Postgres)
        - Password = jobpass
- [ ] Dans le panneau "Query Tool Workspace", dans la zone de requête SQL :
        - GRANT ALL PRIVILEGES ON DATABASE jobdb TO jobuser 
