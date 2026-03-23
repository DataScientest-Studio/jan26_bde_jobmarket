# Jobmarket

**Job Market** est une plateforme **d'analyse du marché du travail en France** construite autour de trois sources de données principales :

1. **France Travail** : API officielle française des offres d'emploi
2. **Welcome to the Jungle** : Scraping de plateforme d'emploi (sitemap + crawling)
3. **Référentiel ROME** : Nomenclature officielle des métiers français

### Objectifs principaux

- 🎯 **Ingérer** les offres d'emploi depuis plusieurs sources (APIs, web scraping)
- 🤖 **Prédire** le code ROME (nomenclature métiers) pour chaque offre
- 💾 **Stocker** les données dans une architecture en médaillon (bronze/silver/gold)
- 🔄 **Fusionner** les données de sources différentes
- 📊 **Monitorer** l'ingestion via dashboards Grafana temps réel
- 📊 **Restituer** des indicateurs sur le marché de l'emploi en France

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Infrastructure Docker](#3-infrastructure-docker)
4. [Pipeline de données](#4-pipeline-de-données)
5. [API FastAPI](#5-api-fastapi)
6. [Modèle ML — Classification ROME](#6-modèle-ml--classification-rome)
7. [Orchestration Airflow](#7-orchestration-airflow)
8. [Base de données PostgreSQL](#8-base-de-données-postgresql)
9. [Stockage objet MinIO](#9-stockage-objet-minio)
10. [Monitoring & Observabilité](#10-monitoring--observabilité)
11. [Configuration](#11-configuration)
12. [Démarrage rapide](#12-démarrage-rapide)

---

## 1. Vue d'ensemble

### Sources de données

| Source | Méthode | Volume estimé | Format brut |
|---|---|---|---|
| **France Travail** | API REST OAuth2 officielle | ~100k offres/cycle | JSONL partitionné |
| **Welcome to the Jungle** | Web scraping (sitemap + HTML) | ~80k URLs | JSONL + HTML (en backup) |

### Flux global

![Schéma général](images/pipeline.png)

### Caractéristiques techniques

- **Architecture médaillon** : Bronze / Silver / Gold (schéma Kimball en étoile)
- **API** : FastAPI, 16 endpoints, tâches asynchrones avec polling
- **ML** : LinearSVC + TF-IDF, ~490 codes ROME, 78% accuracy, Top-5 ≈ 92%
- **Orchestration** : Apache Airflow 3.x, DAG quotidien, notifications Slack
- **Stockage** : MinIO (compatible S3), abstraction Local/S3 interchangeable
- **BDD** : PostgreSQL 15, 3 schémas (public, logs, gold)

---

## 2. Architecture

### Structure du dépôt

```
.
├── dags/                          # DAG Airflow
│   └── jobmarket_daily.py
├── docs/                          # Documentation technique
├── postgres/
│   └── init/                      # Scripts SQL d'initialisation
│       ├── 000_init_user_db.sql
│       ├── 001_job_runs.sql
│       ├── 002_ingestion_logs.sql
│       ├── 003_gold_star_schema.sql
│       └── 004_create_airflow_db.sh
├── src/
│   ├── api/
│   │   ├── main.py                # Application FastAPI
│   │   └── models/                # Schémas Pydantic (request/response)
│   │       ├── base.py            # BaseJobResponse (succès, message, records_count)
│   │       ├── data.py            # Merge, Status, Evolution, StarSchema
│   │       ├── ingest.py          # Ingestion FT, WTTJ
│   │       ├── normalize.py       # Normalisation FT, WTTJ
│   │       └── predict.py         # Prédiction ROME
│   ├── config/
│   │   └── env.py                 # Chargement .env, helpers require_env
│   ├── data/
│   │   └── make_dataset.py        # Création du dataset pour l'entrainement du modède de machine learning
│   ├── ingest/
│   │   ├── bronze/                # Ingestion des sources brutes
│   │   ├── silver/                # Normalisation et fusion
│   │   ├── gold/                  # Chargement star schema
│   │   ├── clients/               # Clients API (FT OAuth2)
│   │   ├── data_models/           # Classes Bronze et Silver
│   │   └── tools/                 # Rate limiter, utilitaires communs
│   ├── models/                    # ML : entraînement et prédiction
│   ├── observability/
│   │   └── job_store.py           # Suivi des tâches d'exécution dans la base Postgres (base jobdb schema public)
│   ├── storage/
│   │   └── storage.py             # Abstraction Local/S3
│   └── utils/                     # Helpers (rome, logging, text, time)
├── grafana/                       # Dashboards et datasources provisionnés
├── pgadmin/                       # Configuration pgAdmin
├── streamlit                      # Code de l'application streamlit
├── docker-compose.yml
├── Dockerfile                     # Image Jupyter
├── Dockerfile.api                 # Image FastAPI
├── Dockerfile.airflow             # Image Airflow 3.x
└── requirements.txt
```

### Couches de données

```
┌─────────────────────────────────────────────────────────────────┐
│  BRONZE — Données brutes                                        │
│  Format : JSONL gzip, partitionné par dt=YYYY-MM-DD / run_id   │
│  Stockage : MinIO  bronze/france_travail/  bronze/wttj/         │
├─────────────────────────────────────────────────────────────────┤
│  SILVER — Données normalisées                                   │
│  Format : Parquet, schéma canonique unique (FT + WTTJ)          │
│  Stockage : MinIO  silver/merged/  silver/status/               │
│  Enrichissement : ROME prédit par ML, statut cycle de vie       │
├─────────────────────────────────────────────────────────────────┤
│  GOLD — Star schema analytique                                  │
│  Format : PostgreSQL (schéma gold), tables Kimball              │
│  Accès : requêtes SQL directes, dashboards Grafana              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Infrastructure Docker

### Services

| Service | Image | Port | CPU | RAM | Rôle |
|---|---|---|---|---|---|
| `jobmarket-postgres` | postgres:15 | 5432 | 2.0 | 4 GB | Base de données principale |
| `jobmarket-minio` | minio/minio | 9000 / 9001 | 0.5 | 1 GB | Stockage objet S3 |
| `jobmarket-minio-init` | minio/mc | — | — | — | Init bucket (one-shot) |
| `jobmarket-api` | Dockerfile.api | 8000 | 4.0 | 8 GB | API FastAPI |
| `jobmarket-airflow` | Dockerfile.airflow | 8080 | 1.0 | 2 GB | Orchestration |
| `jobmarket-grafana` | grafana:10.2.3 | 3000 | 0.5 | 512 MB | Dashboards |
| `jobmarket-pgadmin` | pgadmin4 | 5050 | 0.5 | 512 MB | Admin PostgreSQL |
| `jobmarket-jupyter` | Dockerfile | 8888 | 1.5 | 5 GB | Notebooks exploration |
| `jobmarket-streamlit` | Dockerfile | 8501 | 1 | 1 GB | Application finale |

Tous les services communiquent sur le réseau bridge `jobmarket-net` par leur nom de service (ex. `jobmarket-postgres:5432`, `jobmarket-api:8000`).

### Dockerfiles

#### `Dockerfile.api` — FastAPI
```dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src ./src
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Le répertoire `src/` est monté en bind mount pour le hot reload en développement.

#### `Dockerfile.airflow` — Airflow 3.x
```dockerfile
FROM apache/airflow:slim-latest
RUN AIRFLOW_VERSION=$(python -c "import airflow; print(airflow.__version__)") && \
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    pip install --no-cache-dir \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
        psycopg2-binary asyncpg apache-airflow-providers-http
```

**Choix techniques Airflow liés à Airflow 3.x + PostgreSQL :**
- `psycopg2-binary` : driver PostgreSQL synchrone (l'url rest  `postgresql+psycopg2://` partout dans le projet)
- `asyncpg` : moteur asynchrone requis par Airflow 3.x en interne (même si aucun code asynchrone utilisé dans le projet)
- `apache-airflow[postgres]` -  C'est un extra Airflow qui installe ses propres dépendances PostgreSQL — mais en faisant ça il réinstalle Airflow lui-même avec des versions potentiellement différentes, ce qui casse les entry_points des providers (les plugins Airflow). Avec comme symptôme typique : AttributeError: 'NoneType'.rsplit dans HttpOperator — le provider HTTP est installé mais non reconnu car ses entry_points sont corrompus. Solution retenue : installer psycopg2-binary + asyncpg directement, sans passer par l'extra.
- Le fichier constraint garantit la compatibilité des versions entre providers

### Volumes

| Volume | Monté dans | Contenu |
|---|---|---|
| `jobdb-data` | jobmarket-postgres | Données PostgreSQL persistantes dans le volume système |
| `./data` | jobmarket-minio | Données Bronze/Silver/Gold (bind mount local) |
| `./src` | jobmarket-api | Code source |
| `./logs` | jobmarket-api | Logs rotatifs API et ingestion |
| `./dags` | jobmarket-airflow | Définitions des DAGs |
| `./postgres/init/` | postgres | Scripts SQL d'initialisation |

---

## 4. Pipeline de données

### Bronze — Ingestion

#### France Travail (API OAuth2)

| Module | Rôle |
|---|---|
| `ingest/clients/france_travail_client.py` | Client OAuth2 : token, refresh, retry |
| `ingest/bronze/ingest_france_travail_jobs.py` | Ingestion paginée par code ROME, fenêtres temporelles |
| `ingest/bronze/ingest_france_travail_rome_metiers.py` | Catalogue des ~532 codes ROME |
| `ingest/tools/france_travail_common.py` | Probe total, extract_and_store, rate limiter |

**Particularités :**
- Rate limit : 10 req/s (token bucket)
- Fenêtres temporelles glissantes de 7 jours (max 260 fenêtres) pour contourner la limite de 3150 résultats par requête FT
- Stockage partitionné : `bronze/france_travail/dt=YYYY-MM-DD/run_id=.../rome=.../part-XXXXXX.jsonl`

#### Welcome to the Jungle (Web scraping)

| Module | Rôle |
|---|---|
| `ingest/bronze/ingest_wttj_collect_urls.py` | Parse les sitemaps XML gzippés (~80k URLs) |
| `ingest/bronze/ingest_wttj_jobs.py` | Scraping HTML multi-threadé |
| `ingest/bronze/ingest_wttj_job_opt.py` | Ingestion optimisée (asyncio + batch) |
| `ingest/tools/welcome_to_the_jungle_fetch_opt.py` | Fetch async avec retry et backoff |

**Particularités :**
- Rate limit configurable : `WTTJ_RPS=2`, `WTTJ_BURST=2`
- Modes : `new` (full), `resume` (reprise sur run_id), `incremental`
- Stockage partitionné par `dt`, `run_id`, `segment` (jobs_raw, companies_raw, urls)
- HTML optionnellement stocké pour rejeu (`WTTJ_STORE_HTML=always`)

### Silver — Normalisation

Les deux sources sont normalisées vers un **schéma canonique unique** (`Silver_Datamodel`) :

| Champ | Type | Description |
|---|---|---|
| `id` | str | Identifiant source |
| `source` | str | `FT` ou `WTTJ` |
| `title` | str | Intitulé normalisé |
| `description` | str | Description nettoyée (HTML strippé) |
| `rome_code` | str | Code ROME prédit par ML |
| `rome_label` | str | Libellé ROME |
| `contract_type` | str | Type de contrat normalisé |
| `experience_level` | str | Niveau d'expérience |
| `naf_code` | str | Code NAF entreprise |
| `job_city` / `job_postal_code` | str | Localisation poste |
| `company_name` | str | Entreprise |
| `salary_min` / `salary_max` | float | Salaire (normalisé annuel) |
| `published_at` / `updated_at` / `unpublished_at` | datetime | Cycle de vie |
| `status` | str | `published` ou `archived` |

**Modules Silver :**

| Module | Rôle |
|---|---|
| `normalize_ft_jobs.py` | Lecture JSONL FT → Silver_Datamodel → Parquet |
| `normalize_wttj_jobs.py` | Lecture JSONL WTTJ → Silver_Datamodel → Parquet (+ appel ML) |
| `merge_ft_wttj_datasets.py` | Fusion des deux datasets normalisés, déduplication |
| `calculate_offer_status.py` | Calcul statut cycle de vie par snapshot (published/unknown/unpublished) |
| `generate_status_evolution_datasets.py` | Datasets analytiques d'évolution temporelle |

**`NormalizeResult`** — objet retourné par les normalisations :
```python
class NormalizeResult:
    job_id: str       # identifiant du run
    status: str       # "SUCCESS" | "ERROR"
    dt: str           # date de traitement (YYYY-MM-DD)
    format: str       # "parquet" | "jsonl" | "csv"
    files: list[str]  # clés MinIO produites
    errors: int       # nombre d'offres en erreur
    rows: int         # nombre de lignes produites (pour records_count API)
```

### Gold — Star Schema

Architecture Kimball (schéma plat, pas de snowflake) : 1 JOIN suffit pour tout niveau d'agrégation.

```
                    ┌──────────────┐
                    │ dim_code_rome│
                    │              │
                    └──────┬───────┘
┌──────────────┐           │           ┌───────────────────┐
│   dim_geo    │           │           │  dim_type_contrat │
│ 6329 codes   ├───────────┤           └─────────┬─────────┘
│ postaux      │           │                     │
└──────────────┘    ┌──────┴───────────────────┐ │
                    │   fact_offre_emploi      │─┤
┌──────────────┐    │  (1 ligne par offre)     │ │
│   dim_naf    ├────│  run_id, snapshot_dt     │ │  ┌──────────────┐
│ 732 codes    │    │  source, url, title      │─┘  │dim_experience│
│ 5 niveaux    │    │  salary_min, salary_max  │    │ niveaux      │
└──────────────┘    └──────────────────────────┘    └──────────────┘
```

**Dimensions :**

| Table | Lignes | Source | Particularités |
|---|---|---|---|
| `dim_geo` | 6 329 | INSEE communes | Dept + région + lat/lon dénormalisés |
| `dim_naf` | 732 | INSEE NAF rév.2 | 5 niveaux plats (NIV1→NIV5 + libellés) |
| `dim_code_rome` | ~1200 | France Travail | Code + libellé |
| `dim_type_contrat` | ~10 | Pipeline | CDI, CDD, Freelance, etc. |
| `dim_experience` | ~5 | Pipeline | Niveaux d'expérience normalisés |

**Chargement incrémental :** chaque import est identifié par un `run_id` déterministe. La table `gold.imported_snapshots` enregistre les imports effectués — les snapshots déjà importés sont sautés (idempotent).

---

## 5. API FastAPI

### Format de réponse unifié

Tous les endpoints héritent de `BaseJobResponse` :
```python
class BaseJobResponse(BaseModel):
    success: bool          # True si l'opération a réussi
    message: str           # Message descriptif
    records_count: int     # Nombre d'enregistrements traités (None si N/A)
```

### Endpoints

#### Monitoring

| Méthode | Path | Description |
|---|---|---|
| GET | `/health` | État de l'API + version modèle ML |
| GET | `/jobs` | Liste des runs avec filtre source/status |
| GET | `/jobs/{run_id}` | Détail d'un run |
| GET | `/tasks/{task_id}` | État d'une tâche asynchrone (polling) |
| GET | `/ingest/status` | Tâches d'ingestion actives |
| GET | `/data/status` | Tâches de transformation actives |

#### Extraction — Bronze

| Méthode | Path | Paramètres clés | Description |
|---|---|---|---|
| POST | `/ingest/rome-metiers` | — | Ingestion catalogue ROME complet |
| POST | `/ingest/france-travail-offers` | `background`, `max_rome_codes` | Ingestion offres FT |
| POST | `/ingest/welcome-to-jungle` | `background`, `max_jobs`, `part_size` | Ingestion offres WTTJ |
| POST | `/ingest/welcome-to-the-jungle/sitemaps` | — | Crawl sitemaps WTTJ |
| POST | `/ingest/welcome-to-the-jungle/jobs-optimized` | `background`, `max_jobs`, `mode` | Ingestion WTTJ optimisée |

#### Transformation — Silver

| Méthode | Path | Default `background` | Description |
|---|---|---|---|
| POST | `/data/normalize-wttj-jobs` | `False` | Normalisation WTTJ Bronze → Silver |
| POST | `/data/normalize-ft-jobs` | `False` | Normalisation FT Bronze → Silver |
| POST | `/data/merge-datasets` | `False` | Fusion FT + WTTJ |
| POST | `/data/status-tracking` | `False` | Calcul cycle de vie offres |
| POST | `/data/status-evolution` | `False` | Génération datasets analytics |

#### Load — Gold

| Méthode | Path | Paramètres clés | Description |
|---|---|---|---|
| POST | `/gold/load-geo-dim` | — | Charge `dim_geo` (6329 CP) |
| POST | `/gold/load-naf-dim` | — | Charge `dim_naf` (732 NAF) |
| POST | `/gold/load-star-schema` | `source_mode`, `incremental` | Charge fact + dimensions |

#### Machine Learning

| Méthode | Path | Paramètres | Description |
|---|---|---|---|
| POST | `/predict` | `intitule`, `description`, `competences` | Prédiction code ROME (top-k) |

### Mode background

Tous les endpoints long-running supportent `?background=false` (sync) et `?background=true` (async) :

```
background=false (défaut) :  attend la fin → retourne le résultat complet
background=true            :  démarre la tâche → retourne un task_id immédiatement
                              → polling via GET /tasks/{task_id}
```

### Architecture async/sync des endpoints

Le choix `async def` vs `def` est délibéré :

- **`def` (thread pool)** : pour les endpoints qui font des I/O bloquants ou du CPU intensif (`normalize_*`, `predict`, `ingest_*`). FastAPI les exécute via `anyio.to_thread.run_sync`, libérant la boucle asyncio.
- **`async def` (boucle asyncio)** : uniquement pour les endpoints légers qui n'ont pas de code bloquant.

> **Attention piège :** les endpoints normalize (`def`) appellent `/predict` en interne via `requests.post()`. Si ces endpoints avaient été `async def`, ils auraient bloqué la boucle asyncio, empêchant `/predict` de répondre → deadlock. 
---

## 6. Modèle ML — Classification ROME

### Objectif

Prédire automatiquement le code ROME (Répertoire Opérationnel des Métiers et des Emplois) d'une offre d'emploi à partir de son titre et de sa description.

### Architecture

```
Texte (titre + description) → TF-IDF Vectorizer → LinearSVC → Top-K prédictions ROME
```

| Composant | Choix | Justification |
|---|---|---|
| Vectorisation | TF-IDF (unigrams + bigrams) | Léger, efficace sur texte court/moyen |
| Classifieur | LinearSVC | Excellentes performances sur classification multi-classe sparse, très rapide à l'inférence |
| Features | `max_features=200000`, `min_df=5`, `sublinear_tf=True` | Équilibre vocabulaire/bruit |

### Performances

| Métrique | Valeur |
|---|---|
| Accuracy Top-1 | ~78% |
| F1 Macro | ~67% (classes rares plus difficiles) |
| Accuracy Top-3 | ~89% |
| Accuracy Top-5 | ~92% |
| Classes | ~490 codes ROME |
| Dataset | ~307k offres d'emploi français |
| Split | 80% train / 10% val / 10% test |

### Stockage du modèle

Le modèle est sérialisé avec `joblib` et stocké dans MinIO :
```
models/rome_tfidf/v2/
├── model.pkl          # Pipeline sklearn (TF-IDF + LinearSVC)
├── label_encoder.pkl  # LabelEncoder (int → code ROME)
└── metadata.json      # version, date, métriques
```

### Lazy loading et cold start

Le modèle est chargé **en mémoire au premier appel** `/predict` (pas au démarrage de l'API) via `_ensure_model_loaded()` avec double-checked locking thread-safe.

Le timeout client est splitté pour absorber ce cold start :
```
ML_CONNECT_TIMEOUT = 10s   # délai connexion TCP — court (échec rapide si service mort)
ML_READ_TIMEOUT    = 120s  # délai réponse — long (absorbe le chargement du modèle)
```

En cas d'erreur ML (timeout, service indisponible, erreur HTTP), la normalisation continue avec `rome_code=None` — le pipeline ne s'interrompt pas.

### Hyperparamètre tuning

4 stratégies disponibles via `TUNING_STRATEGY` :
- `none` : valeurs par défaut
- `manual` : grille manuelle sur C, ngram_max, min_df
- `grid` : GridSearchCV exhaustif (lent)
- `random` : RandomizedSearchCV (compromis)

---

## 7. Orchestration Airflow

### DAG `jobmarket_daily`

- **Schedule :** quotidien à 3h00 UTC (`0 3 * * *`)
- **Retries :** 1 retry avec délai de 5 min
- **Mode debug :** déclenchable manuellement avec `{"debug": true}` (FT limité à 2 codes ROME, WTTJ à 50 offres)

### Graphe de dépendances

```
notify_start
     │
     ├─── resolve_ft_endpoint ──► ingest_ft ──────────────┐
     │                                                     │
     └─── resolve_wttj_endpoint ─► ingest_wttj ───────────┤
                                                           │
                            Bronze → Silver ───────────────┤
                                                           ▼
                                          ┌─── normalize_wttj ───┐
                                          └─── normalize_ft ─────┤
                                                                  │
                                                                merge
                                                                  │
                                                          status_tracking
                                                                  │
                                                          status_evolution
                                                                  │
                            Silver → Gold ──────────────────────────►
                                                          load_star_schema
```

### Tâches

| Task | Opérateur | Endpoint | Timeout |
|---|---|---|---|
| `notify_start` | PythonOperator | — | — |
| `resolve_ft_endpoint` | PythonOperator | — | — |
| `resolve_wttj_endpoint` | PythonOperator | — | — |
| `ingest_ft` | HttpOperator | `POST /ingest/france-travail-offers` | 2h |
| `ingest_wttj` | HttpOperator | `POST /ingest/welcome-to-jungle` | 12h |
| `normalize_wttj` | HttpOperator | `POST /data/normalize-wttj-jobs` | 1h |
| `normalize_ft` | HttpOperator | `POST /data/normalize-ft-jobs` | 1h |
| `merge` | HttpOperator | `POST /data/merge-datasets` | 30 min |
| `status_tracking` | HttpOperator | `POST /data/status-tracking` | 30 min |
| `status_evolution` | HttpOperator | `POST /data/status-evolution` | 30 min |
| `load_star_schema` | HttpOperator | `POST /gold/load-star-schema?source_mode=auto` | 30 min |

### Connexion Airflow → API

La connexion est déclarée dans Airflow comme une connexion de type `http` nommée `jobmarket_api` (host = `jobmarket-api`, port = `8000`). Elle est créée ou recrée à chaque démarrage du conteneur Airflow :

```bash
airflow connections delete jobmarket_api 2>/dev/null
airflow connections add jobmarket_api --conn-type http --conn-host jobmarket-api --conn-port 8000
```

> Le `delete` avant le `add` est obligatoire. Si la connexion existe avec `conn_type=generic` (créée avant l'installation du provider HTTP), `HttpOperator` échoue avec `AttributeError: 'NoneType'.rsplit`.

### Notifications Slack

Les callbacks suivants envoient des messages sur `SLACK_WEBHOOK_URL` (si définie) :
- `notify_start` : démarrage du DAG
- `_on_task_success` : succès de chaque tâche (avec `records_count` extrait de XCom)
- `_on_failure` : échec d'une tâche
- `_on_dag_success` : fin réussie du DAG complet

Si `SLACK_WEBHOOK_URL` n'est pas définie, les callbacks sont silencieux (pas d'erreur).

---

## 8. Base de données PostgreSQL

### Schémas

| Schéma | Tables | Rôle |
|---|---|---|
| `public` | `job_runs`, `ingestion_logs` | Suivi opérationnel |
| `gold` | `fact_offre_emploi`, `dim_*`, `stg_offer`, `imported_snapshots` | Star schema analytique |
| `airflow` | (interne Airflow) | Métadonnées DAG, XCom, connections |

### `job_runs` — Tracking des runs longs

```sql
CREATE TABLE job_runs (
    run_id          TEXT PRIMARY KEY,
    job_type        TEXT,          -- "ingest" | "data" | "gold"
    source          TEXT,          -- "france_travail" | "wttj" | ...
    status          TEXT,          -- "running" | "success" | "failed"
    started_at      TIMESTAMPTZ,
    ended_at        TIMESTAMPTZ,
    duration_ms     BIGINT,
    progress_pct    INT,
    records_count   BIGINT,
    errors_count    BIGINT,
    params_json     JSONB,
    result_json     JSONB,
    error_text      TEXT,
    updated_at      TIMESTAMPTZ
);
```

Exposé via `GET /jobs` et `GET /jobs/{run_id}`.

### `ingestion_logs` — Logs centralisés

```sql
CREATE TABLE ingestion_logs (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ,
    endpoint        VARCHAR(50),
    level           VARCHAR(10),   -- INFO | WARNING | ERROR
    task_id         VARCHAR(100),
    message         TEXT,
    duration_sec    NUMERIC,
    records_count   BIGINT,
    extra_metadata  JSONB
);
```

Auto-purge des logs de plus de 30 jours (trigger PostgreSQL).

### Star Schema Gold

```sql
-- Fait principal
CREATE TABLE gold.fact_offre_emploi (
    offer_id            TEXT,
    run_id              TEXT,
    snapshot_dt         DATE,
    source              TEXT,
    url                 TEXT,
    title               TEXT,
    salary_min          NUMERIC,
    salary_max          NUMERIC,
    published_at        TIMESTAMPTZ,
    rome_key            INTEGER REFERENCES gold.dim_code_rome,
    geo_key             INTEGER REFERENCES gold.dim_geo,
    naf_key             INTEGER REFERENCES gold.dim_naf,
    contrat_key         INTEGER REFERENCES gold.dim_type_contrat,
    experience_key      INTEGER REFERENCES gold.dim_experience
);

-- Tracking imports incrémentaux
CREATE TABLE gold.imported_snapshots (
    run_id      TEXT PRIMARY KEY,
    snapshot_dt DATE,
    source      TEXT,
    import_ts   TIMESTAMPTZ DEFAULT now()
);
```

---

## 9. Stockage objet MinIO

### Bucket `jobmarket` — Organisation

```
jobmarket/
├── bronze/
│   ├── france_travail/
│   │   ├── rome/                           # Catalogue ROME
│   │   └── offers/
│   │       └── dt=YYYY-MM-DD/
│   │           └── run_id=.../
│   │               └── rome=MXXXX/
│   │                   └── part-XXXXXX.jsonl
│   └── wttj/
│       └── dt=YYYY-MM-DD/
│           └── run_id=.../
│               └── segment=jobs_raw/
│                   └── part-XXXXXX.jsonl
├── silver/
│   ├── normalized/                         # FT et WTTJ normalisés séparément
│   ├── merged/                             # Dataset fusionné
│   └── status/                             # Datasets cycle de vie
├── gold/
│   └── datasets/                           # Exports analytiques optionnels
├── models/
│   └── rome_tfidf/v2/                      # Artefacts ML
└── insee/
    ├── int_courts_naf_rev_2.csv             # Nomenclature NAF
    └── 20230823-communes-departement-region.csv  # Géo communes
```

### Abstraction Storage

L'interface `Storage` est interchangeable entre Local et S3 via `STORAGE_BACKEND` :

```python
# Lecture
data = storage.read_bytes("bronze/france_travail/...")
df   = storage.read_parquet("silver/merged/...")
records = list(storage.read_jsonl("bronze/wttj/..."))

# Écriture
storage.write_parquet("silver/merged/...", df)
storage.write_jsonl("bronze/wttj/...", records)

# Listing
keys = list(storage.list_keys("bronze/france_travail/dt=2026-03-18/"))
```

---

## 10. Monitoring & Observabilité

### Niveaux de logging

| Niveau | Destination | Format | Usage |
|---|---|---|---|
| Opérationnel | Console + fichier rotatif | Texte lisible | Debug, suivi |
| Structuré | Fichier JSONL optionnel | JSON par ligne | Grafana / Loki |
| Persisté | PostgreSQL `ingestion_logs` | JSONB | Historique, alertes |

### JobStore

`src/observability/job_store.py` — tracking temps réel des jobs longs dans PostgreSQL :

```python
class JobStore:
    def create(run_id, job_type, source, params, message)
    def progress(run_id, progress_pct, message, records_count)
    def finish(run_id, status, result, error_text)
    def get_run(run_id) -> JobRun
    def list_runs(source, status, limit) -> list[JobRun]
```

Graceful degradation : fonctionne sans PostgreSQL (pas d'erreur si DSN absent).

### Grafana

- Accessible sur [http://localhost:3000](http://localhost:3000)
- Datasource PostgreSQL provisionnée automatiquement
- Dashboards provisionés depuis `grafana/dashboards/`
- Credentials : `admin` / `admin` (configurable via `.env`)

---

## 11. Configuration

### Variables d'environnement principales

#### France Travail

| Variable | Défaut | Description |
|---|---|---|
| `API_KEY` | — | Clé OAuth2 FT (obligatoire) |
| `API_SECRET` | — | Secret OAuth2 FT (obligatoire) |
| `FT_RATE_LIMIT_RPS` | `10` | Requêtes par seconde |
| `FT_WINDOW_DAYS` | `7` | Durée fenêtre temporelle |
| `FT_MAX_ROME_CODES` | `0` | 0 = tous les codes |
| `FT_MAX_RETRIEVABLE` | `3150` | Limite API FT par requête |

#### Welcome to the Jungle

| Variable | Défaut | Description |
|---|---|---|
| `WTTJ_RUN_MODE` | `new` | `new` / `resume` / `incremental` |
| `WTTJ_WORKERS` | `6` | Threads de scraping |
| `WTTJ_RPS` | `2` | Requêtes par seconde |
| `WTTJ_PART_SIZE` | `5000` | Taille chunk JSONL |
| `WTTJ_MAX_JOBS` | `0` | 0 = toutes les offres |

#### Stockage

| Variable | Défaut | Description |
|---|---|---|
| `STORAGE_BACKEND` | `S3` | `local` ou `s3` |
| `S3_ENDPOINT_URL` | `http://localhost:9000` | URL MinIO |
| `S3_BUCKET` | `jobmarket` | Nom du bucket |
| `S3_ACCESS_KEY` | `minioadmin` | Clé MinIO |
| `S3_SECRET_KEY` | `minioadmin123` | Secret MinIO |

#### Machine Learning

| Variable | Défaut | Description |
|---|---|---|
| `ML_HOST_API` | `http://localhost:8000` | URL API ML |
| `ML_ENDPOINT` | `predict` | Path du endpoint |
| `ML_CONNECT_TIMEOUT` | `10` | Timeout connexion TCP (s) |
| `ML_READ_TIMEOUT` | `120` | Timeout réponse (s, absorbe cold start) |
| `MODEL_NAME` | `rome_tfidf` | Nom du modèle |
| `MODEL_VERSION` | `v2` | Version du modèle |

#### Airflow

| Variable | Défaut | Description |
|---|---|---|
| `AIRFLOW_SECRET_KEY` | — | Clé de chiffrement Airflow (obligatoire) |
| `AIRFLOW_ADMIN_USER` | `admin` | Utilisateur UI Airflow |
| `AIRFLOW_ADMIN_PASSWORD` | `admin` | Mot de passe UI Airflow |
| `SLACK_WEBHOOK_URL` | — | Webhook Slack (optionnel) |

#### PostgreSQL

| Variable | Défaut | Description |
|---|---|---|
| `POSTGRES_DB` | `jobdb` | Nom de la base |
| `POSTGRES_USER` | `jobuser` | Utilisateur |
| `POSTGRES_PASSWORD` | `jobpass` | Mot de passe |
| `JOBSTORE_DSN` | — | DSN complet pour JobStore |

> Voir `.env.example` pour la liste complète des variables.

> **Reload des variables d'env :** après modification du `.env`, les conteneurs doivent être recrées (`docker compose up -d --force-recreate <service>`) — un simple restart ne recharge pas les variables d'environnement injectées par Docker Compose.

---

## 12. Démarrage rapide

### Prérequis

- Docker >= 24.0
- Docker Compose >= 2.20
- Python 3.11
- `pip` >= 23.0 et `venv` (inclus avec Python 3.11)
- 4GB RAM minimum, 8GB recommandé
- 50GB disque (données + modèles ML)

### Installation locale

```bash
# 1. Clone repository
git clone https://github.com/...  /jan26_bde_jobmarket
cd jan26_bde_jobmarket

# 2. Variables d'environnement
cp .env.example .env              # Adapter API_KEY, API_SECRET, etc

# Activation
python -m venv env_job_market
source env_job_market/bin/activate  # Unix
# ou
env_job_market\Scripts\Activate.ps1  # Windows

# 3. Installation des dépendances
pip install -r requirements.txt

# 3.2 Mise en place de la configuration pgadmin
copier et renommer le fichier /pgadmin/servers.json.example en /pgadmin/servers.json en indiquant le mot de passe défini pour POSTGRES_PASSWORD dans le .env

# 4. Lancement des services Docker 
docker-compose up -d               # Postgres, MinIO, Grafana, API

# En cas de modifications des variables dans le env il faut recréer le container:
docker-compose up -d --force-recreate jobmarket-api

# 5. Verifier le bon lancement des services
docker-compose ps                 
curl http://localhost:8000/docs   # FastAPI Swagger
curl http://localhost:3000/ # Dashboard Grafana
curl http://localhost:8888/ # Jupyter (token=jobmarket)

# 6. Reconstruction du schema gold si volume docker existant

# Création schéma Gold
docker exec jobmarket-postgres psql -U jobuser -d jobdb -f /docker-entrypoint-initdb.d/003_gold_star_schema.sql

# Alimentation des codes postaux (dim_geo)
env_job_market\Scripts\python.exe -m src.ingest.gold.load_geo_dim
# Alimentation des codes naf (dim_naf)
env_job_market\Scripts\python.exe -m src.ingest.gold.load_naf_dim
# Alimentation du schema gold (wip) à partir des dataset d'historique de statut

env_job_market\Scripts\python.exe -m src.ingest.gold.load_star_schema --source-mode auto



```

### Accès aux services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API FastAPI** | http://localhost:8000 | - |
| **Swagger UI** | http://localhost:8000/docs | - |
| **Grafana** | http://localhost:3000 | admin / admin |
| **pgAdmin** | http://localhost:5050 | jobuser / jobpass |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |
| **Jupyter Notebook** | http://localhost:8888 | token : jobmarket |
| **Streamlit** | http://localhost:8501 | - |

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

**Installation**
- [ ] Cloner le repository
- [ ] Copier `.env.example` → `.env` et configurer `API_KEY`, `API_SECRET`
- [ ] Créer le venv : `python -m venv env_job_market`
- [ ] Activer le venv et installer les dépendances : `pip install -r requirements.txt`
- [ ] Copier `pgadmin/servers.json.example` → `pgadmin/servers.json`

**Démarrage**
- [ ] `docker-compose up -d`
- [ ] Vérifier que tous les services sont `Up` : `docker-compose ps`
- [ ] Tester l'API : `curl http://localhost:8000/docs`

**Vérification pgAdmin** (`http://localhost:5050`)
- [ ] Créer une connexion serveur : host=`jobmarket-postgres`, port=`5432`, db=`jobdb`, user=`jobuser`
- [ ] Vérifier la connexion — en cas d'échec, consulter les logs : `docker logs jobmarket-postgres`
- [ ] Exécuter dans le Query Tool : `GRANT ALL PRIVILEGES ON DATABASE jobdb TO jobuser`

**Tests fonctionnels**
- [ ] Lancer une ingestion test (ROME codes) : `POST /ingest/rome-metiers?background=true`
- [ ] Tester la prédiction ROME : `POST /predict`
- [ ] Exécuter une ingestion FT/WTTJ en mode debug (2 codes ROME)
- [ ] Vérifier les dashboards Grafana : `http://localhost:3000`

**Production**
- [ ] Changer tous les mots de passe par défaut (`.env`)
- [ ] Configurer les backups PostgreSQL
