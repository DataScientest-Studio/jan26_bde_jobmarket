# Organisation des Logs par Endpoint

## Structure des dossiers

Les logs sont organisés par catégorie d'opération pour une meilleure observabilité :

```
logs/
├── api/
│   ├── main.log              # Logs principaux de l'API
│   ├── errors.log            # Erreurs globales uniquement
│   └── structured.jsonl      # Logs JSON structurés (si ENABLE_GRAFANA_LOGS=true)
├── ingestion/
│   ├── rome_metiers.log      # Ingestion des codes ROME
│   ├── france_travail_offers.log  # Ingestion des offres France Travail
│   ├── wttj.log              # Ingestion Welcome to the Jungle
│   └── merge_datasets.log    # Fusion des datasets FT + WTTJ
└── prediction/
    └── rome_prediction.log   # Prédictions de codes ROME
```

## Configuration

Les logs sont configurés via variables d'environnement :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `LOG_LEVEL` | `INFO` | Niveau de log (DEBUG, INFO, WARNING, ERROR) |
| `LOG_MAX_BYTES` | `10485760` | Taille max d'un fichier (10MB par défaut) |
| `LOG_BACKUP_COUNT` | `5` | Nombre de fichiers de backup gardés |
| `ENABLE_GRAFANA_LOGS` | `false` | Active les logs JSON pour Grafana/Loki |

## Caractéristiques

### Rotation automatique
- Les fichiers logs ont une taille maximale (10MB par défaut)
- Rotation automatique avec conservation de 5 backups
- Format : `fichier.log`, `fichier.log.1`, `fichier.log.2`, etc.

### Logs séparés par endpoint
- Chaque type d'opération a son propre fichier
- Facilite le débogage et l'analyse
- Pas de mélange entre ingestions et prédictions

### Double sortie
- Console : pour le débogage en temps réel
- Fichiers : pour la persistance et l'analyse

### Logs structurés optionnels
- Activez `ENABLE_GRAFANA_LOGS=true` pour logs JSON
- Compatible avec Grafana/Loki pour visualisation avancée
- Format JSONL (une ligne JSON par log)

**⚠️ Important :** Les logs JSON structurés sont générés uniquement pour :
- Les opérations en **mode background** (`?background=true`)
- Les événements système (démarrage, chargement du modèle, etc.)

Les exécutions **synchrones** (`?background=false`) génèrent uniquement des logs texte dans les fichiers `.log` standards.

## Commandes utiles

### Consulter les logs en temps réel

```powershell
# Logs de l'API principale
Get-Content logs/api/main.log -Wait -Tail 20

# Logs d'ingestion ROME
Get-Content logs/ingestion/rome_metiers.log -Wait -Tail 20

# Logs de prédiction
Get-Content logs/prediction/rome_prediction.log -Wait -Tail 20

# Toutes les erreurs
Get-Content logs/api/errors.log -Wait -Tail 20
```

### Rechercher dans les logs

```powershell
# Rechercher une erreur spécifique
Select-String -Path logs/**/*.log -Pattern "ERROR"

# Rechercher toutes les ingestions ROME
Select-String -Path logs/ingestion/rome_metiers.log -Pattern "Ingestion"

# Compter les prédictions réussies
(Select-String -Path logs/prediction/rome_prediction.log -Pattern "réussie").Count
```

### Analyser les logs

```powershell
# Voir les 100 dernières lignes
Get-Content logs/api/main.log -Tail 100

# Logs d'aujourd'hui seulement
Get-Content logs/api/main.log | Select-String (Get-Date -Format "yyyy-MM-dd")

# Taille totale des logs
Get-ChildItem -Path logs -Recurse -File | Measure-Object -Property Length -Sum
```

## Exemple de contenu

### logs/ingestion/rome_metiers.log
```
2026-02-25 15:08:16 - INFO - 📥 Début de l'ingestion synchrone des codes ROME
2026-02-25 15:08:16 - INFO - ✅ Ingestion réussie: 1584 codes ROME
```

### logs/prediction/rome_prediction.log
```
2026-02-25 15:08:53 - INFO - Requête de prédiction - Intitulé: Data Scientist
2026-02-25 15:08:53 - INFO - Prédiction réussie - Code ROME: M1405
```

### logs/api/main.log
```
2026-02-25 15:07:50 - src.api.main - INFO - 📝 Logging configuré - Niveau: INFO, Grafana: False
2026-02-25 15:08:03 - src.api.main - INFO - ✅ Modèle chargé avec succès: rome_tfidf vv2
```

## Monitoring avec Grafana (optionnel)

Pour activer le monitoring avancé avec Grafana + Loki + Promtail :

### 1. Configuration

Activez les logs structurés dans `.env` :
```bash
ENABLE_GRAFANA_LOGS=true
```

### 2. Démarrage des services

Démarrez tous les services incluant le stack de monitoring :
```powershell
# Démarrer tous les services (API + MinIO + Grafana + Loki + Promtail)
docker compose up -d

# Vérifier que tous les services sont actifs
docker compose ps
```

### 3. Accès aux interfaces

| Service | URL | Identifiants |
|---------|-----|--------------|
| Grafana | http://localhost:3000 | admin / jobmarket2026 (par défaut, configurable dans .env) |
| Loki API | http://localhost:3100 | - |

### 4. Utilisation de Grafana

1. Connectez-vous à Grafana (admin/jobmarket2026 ou vos identifiants du .env)
2. **Dashboards disponibles:**
   - **Job Market API - Monitoring** : Vue d'ensemble globale
   - **Job Market - Ingestion Temps Réel** : ⭐ **Recommandé pour le suivi des ingestions**
3. Accès rapide:
   - Dashboard global: http://localhost:3000/d/jobmarket-api-monitoring
   - Dashboard temps réel: http://localhost:3000/d/jobmarket-ingestion-realtime

**Dashboard "Ingestion Temps Réel" - Fonctionnalités:**
- 📊 **Table historique** : Toutes les ingestions avec statut, durée, volume
- 📈 **Métriques 24h** : Succès, échecs, durée moyenne, total records
- ⏱️ **Percentile 95** : Identifier les ingestions lentes
- 📦 **Volume par type** : Rome métiers, offres FT, WTTJ
- 🔄 **Refresh 5s** : Mise à jour automatique en temps réel
- 📋 **Logs live** : Flux des logs d'ingestion en cours

**Suivi individuel d'une tâche:**
```logql
# Dans Explore ou le panel de logs
{log_type="structured"} | json | task_id="rome-metiers-20260225T150816"

# Voir toutes les étapes d'une ingestion FT
{component="ingestion"} |= "rome-metiers" | json
```

**Métriques agrégées:**
```logql
# Total d'offres ingérées sur 24h
sum(sum_over_time({event_type="ingestion_completed"} | json | unwrap records_count [24h]))

# Taux de succès
sum(count_over_time({event_type="ingestion_completed"}[1h])) 
/ 
sum(count_over_time({event_type=~"ingestion_completed|ingestion_failed"}[1h])) * 100

# Durée p95 par type
quantile_over_time(0.95, {event_type="ingestion_completed"} | json | unwrap duration_sec [1h]) by (task_type)
```

Pour créer vos propres requêtes, allez dans **Explore** et utilisez LogQL :

```logql
# Tous les logs de l'API
{job="jobmarket-api"}

# Logs structurés uniquement (opérations background)
{job="jobmarket-api", log_type="structured"}

# Logs d'erreurs
{job="jobmarket-api", level="ERROR"}

# Logs d'un endpoint spécifique
{job="jobmarket-api", component="ingestion"}

# Logs d'ingestion ROME
{job="jobmarket-api"} |= "rome_metiers"

# Logs avec un task_id particulier (uniquement tasks background)
{job="jobmarket-api"} | json | task_id="rome-metiers-20260225T150816"

# Statistiques: nombre de logs par event_type
sum by (event_type) (count_over_time({job="jobmarket-api", log_type="structured"}[1h]))

# Ingestions complétées (uniquement background)
{log_type="structured", event_type="ingestion_completed"}

# Durée moyenne des ingestions background
avg_over_time({event_type="ingestion_completed"} | json | unwrap duration_sec [1h])
```

### 5. Créer un dashboard

Exemples de panels utiles :

**Panel 1 - Taux de succès des ingestions**
```logql
sum by (status) (count_over_time({log_type="structured", event_type="ingestion_completed"}[1h]))
```

**Panel 2 - Temps d'exécution moyen**
```logql
avg(duration_sec) by (task_type) from {log_type="structured", event_type="ingestion_completed"}
```

**Panel 3 - Erreurs par type**
```logql
sum by (error_type) (count_over_time({log_type="structured", event_type=~".*error|.*failed"}[1h]))
```

### 6. Arrêter les services de monitoring

```powershell
# Arrêter seulement Grafana/Loki/Promtail
docker compose stop grafana loki promtail

# Arrêter tous les services
docker compose down

# Supprimer aussi les volumes (⚠️ perd les dashboards Grafana)
docker compose down -v
```

### Architecture du monitoring

```
┌─────────────┐
│   API       │
│  FastAPI    │──writes──▶ logs/api/structured.jsonl
└─────────────┘           logs/api/main.log
                         logs/ingestion/*.log
                         logs/prediction/*.log
                                │
                                │ reads
                                ▼
                         ┌─────────────┐
                         │  Promtail   │
                         │  (collecte) │
                         └─────────────┘
                                │
                                │ push
                                ▼
                         ┌─────────────┐
                         │    Loki     │
                         │  (agrégation)│
                         └─────────────┘
                                │
                                │ query
                                ▼
                         ┌─────────────┐
                         │   Grafana   │──http://localhost:3000
                         │   (visuels) │
                         └─────────────┘
```

### Fichiers de configuration

Les configurations du stack sont dans le dossier `monitoring/` :

- `loki-config.yaml` : Configuration de Loki (storage, limites)
- `promtail-config.yaml` : Jobs de collecte des logs
- `grafana-datasources.yaml` : Configuration auto de la datasource Loki
