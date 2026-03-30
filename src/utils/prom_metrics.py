"""
Métriques Prometheus pour le projet jobmarket.

Deux catégories :
- Métriques FastAPI (registry global) : exposées sur /metrics, scrape direct par Prometheus.
- Métriques batch (registry isolé)    : poussées vers le Pushgateway depuis les scripts
  d'ingestion éphémères (load_star_schema, DAGs Airflow).

Usage FastAPI (dans main.py) :
    from src.utils.prom_metrics import ML_PREDICTIONS_TOTAL, ML_COLD_START_SECONDS

Usage batch (dans load_star_schema.py) :
    from src.utils.prom_metrics import push_pipeline_metrics
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Métriques FastAPI — registry global (exposées via /metrics)
# -----------------------------------------------------------------------

ML_PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Nombre total de prédictions ROME effectuées",
)

ML_COLD_START_SECONDS = Histogram(
    "ml_cold_start_seconds",
    "Durée du chargement du modèle ML au premier appel /predict (cold start)",
    buckets=[1, 2, 5, 10, 20, 30, 60],
)

# -----------------------------------------------------------------------
# Helper push Pushgateway — registry isolé par appel
# -----------------------------------------------------------------------

def push_pipeline_metrics(
    run_id: str,
    stage: str,
    source: str,
    duration_seconds: float,
    status: int,
    rows: Optional[int] = None,
    errors: Optional[int] = None,
    # Gold-only
    fact_count: Optional[int] = None,
    company_unknown_ratio: Optional[float] = None,
    rome_unknown_ratio: Optional[float] = None,
    # Bronze-only
    api_calls: Optional[int] = None,
    # Silver status_tracking-only
    status_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """Pousse les métriques d'un run pipeline vers le Pushgateway.

    Silencieux en cas d'échec (le Pushgateway est optionnel — ne doit pas
    bloquer le pipeline si le service est indisponible).

    Args:
        run_id: Identifiant du run.
        stage: Couche médaillon (bronze, silver, gold).
        source: Source des données (france_travail, wttj, merged, status_analytics).
        duration_seconds: Durée totale du run en secondes.
        status: 1 = succès, 0 = échec.
        rows: Nb lignes traitées / ingérées.
        errors: Nb erreurs rencontrées.
        fact_count: [gold] Nb total de lignes dans fact_offre_emploi.
        company_unknown_ratio: [gold] Ratio offres sans entreprise identifiée [0.0-1.0].
        rome_unknown_ratio: [gold] Ratio offres sans code ROME [0.0-1.0].
        api_calls: [bronze] Nb d'appels API effectués.
        status_counts: [silver/status_tracking] Distribution des statuts d'offres par source.
            Format : {"FT": {"published": N, "unpublished": N, "reappeared": N}, "WTTJ": {...}}
    """
    pushgateway_url = os.getenv("PUSHGATEWAY_URL", "http://jobmarket-pushgateway:9091")

    registry = CollectorRegistry()
    labels = {"stage": stage, "source": source, "run_id": run_id}
    label_keys = list(labels.keys())

    Gauge("pipeline_run_status", "Statut du run (1=succès, 0=échec)", label_keys, registry=registry).labels(**labels).set(status)
    Gauge("pipeline_last_run_unixtime", "Timestamp Unix (millisecondes) du dernier run", label_keys, registry=registry).labels(**labels).set(int(time.time() * 1000))
    Gauge("pipeline_run_duration_seconds", "Durée du run en secondes", label_keys, registry=registry).labels(**labels).set(duration_seconds)

    if rows is not None:
        Gauge("pipeline_rows_total", "Nb lignes traitées / ingérées", label_keys, registry=registry).labels(**labels).set(rows)

    if errors is not None:
        Gauge("pipeline_errors_total", "Nb erreurs rencontrées", label_keys, registry=registry).labels(**labels).set(errors)

    if api_calls is not None:
        Gauge("pipeline_api_calls_total", "Nb appels API effectués (bronze)", label_keys, registry=registry).labels(**labels).set(api_calls)

    if fact_count is not None:
        Gauge("data_fact_table_size", "Nb total de lignes dans fact_offre_emploi", registry=registry).set(fact_count)

    if company_unknown_ratio is not None:
        Gauge("data_company_unknown_ratio", "Ratio offres sans entreprise identifiée", registry=registry).set(company_unknown_ratio)

    if rome_unknown_ratio is not None:
        Gauge("data_rome_unknown_ratio", "Ratio offres sans code ROME identifié", registry=registry).set(rome_unknown_ratio)

    if status_counts is not None:
        status_gauge = Gauge(
            "pipeline_status_count",
            "Nb d'offres par statut de cycle de vie (published/unpublished/reappeared)",
            ["stage", "offer_source", "status"],
            registry=registry,
        )
        for offer_source, counts in status_counts.items():
            if not isinstance(counts, dict):
                continue
            for status_label, count in counts.items():
                if not isinstance(count, (int, float)):
                    continue
                status_gauge.labels(stage=stage, offer_source=offer_source, status=status_label).set(count)

    try:
        push_to_gateway(pushgateway_url, job=f"jobmarket_{stage}_{source}", registry=registry)
        logger.info("Prometheus metrics pushed to %s (stage=%s source=%s run_id=%s)", pushgateway_url, stage, source, run_id)
    except Exception as exc:
        logger.warning("Failed to push metrics to Pushgateway (%s): %s", pushgateway_url, exc)
