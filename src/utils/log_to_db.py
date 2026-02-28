"""
Utility to log ingestion events directly to PostgreSQL database.

This module provides a simple interface to log messages with metadata.
It uses psycopg for database interactions, but degrades gracefully if psycopg is not installed.
"""
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Graceful degradation
# If psycopg is not installed, 
# log_to_db will fallback to console logging
# but the code can still run without errors
try:
    import psycopg
    from psycopg.types.json import Json
    from psycopg_pool import ConnectionPool
except Exception:  # pragma: no cover - optional dependency
    psycopg = None
    Json = None
    ConnectionPool = None

# Logger de fallback pour les erreurs de logging en DB
logger = logging.getLogger(__name__)

# Pool de connexions PostgreSQL (réutilisation efficace)
_db_pool: Optional[ConnectionPool] = None


def get_db_pool() -> ConnectionPool:
    """Récupère ou crée le pool de connexions PostgreSQL"""
    global _db_pool
    
    if psycopg is None or ConnectionPool is None:
        raise RuntimeError("psycopg or psycopg_pool not installed")
    
    if _db_pool is None:
        dsn = os.getenv('JOBSTORE_DSN')
        if not dsn:
            raise ValueError("JOBSTORE_DSN environment variable not set")
        
        # Pool avec min 1, max 10 connexions
        _db_pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=10,
            open=True
        )
        logger.info("📊 Pool de connexions ingestion_logs créé")
    
    return _db_pool


def log_to_db(
    endpoint: str,
    level: str,
    message: str,
    task_id: Optional[str] = None,
    duration_sec: Optional[float] = None,
    records_count: Optional[int] = None,
    error_count: Optional[int] = None,
    **extra_metadata
) -> bool:
    """
    Enregistre un log d'ingestion directement en PostgreSQL.
    
    Args:
        endpoint: Nom de l'endpoint (ex: 'rome_metiers', 'france_travail_offers')
        level: Niveau de log ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        message: Message du log
        task_id: Identifiant de la tâche (optionnel)
        duration_sec: Durée de l'opération en secondes (optionnel)
        records_count: Nombre d'enregistrements traités (optionnel)
        error_count: Nombre d'erreurs rencontrées (optionnel)
        **extra_metadata: Métadonnées additionnelles stockées en JSONB (ex: ETA, source, file, etc)
    
    Returns:
        True si le log a été enregistré avec succès, False sinon
    
    Example:
        >>> log_to_db(
        ...     'normalize_wttj_jobs',
        ...     'INFO',
        ...     'Processing part-000800.jsonl | 800/10000 records processed so far | ETA: 00:12:34',
        ...     task_id='wttj-normalize-20260228-xxxx',
        ...     records_count=800,
        ...     ETA='00:12:34',
        ...     file='part-000800.jsonl',
        ...     status='RUNNING'
        ... )
        True
    """
    try:
        # Vérification de disponibilité de psycopg
        if psycopg is None or ConnectionPool is None:
            logger.warning("⚠️ psycopg non disponible, fallback vers console")
            logger.info(f"[FALLBACK] {endpoint} | {level} | {message}")
            return False
        
        pool = get_db_pool()
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Construction de la requête INSERT
                cur.execute(
                    """
                    INSERT INTO ingestion_logs (
                        timestamp, endpoint, level, task_id, message,
                        duration_sec, records_count, error_count, extra_metadata
                    ) VALUES (
                        NOW(), %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        endpoint,
                        level.upper(),
                        task_id,
                        message,
                        duration_sec,
                        records_count,
                        error_count if error_count is not None else 0,
                        Json(extra_metadata) if extra_metadata else None
                    )
                )
        return True
    
    except Exception as e:
        # Fallback sur la console en cas d'erreur DB
        logger.error(f"❌ Échec log_to_db pour {endpoint}: {e}")
        logger.info(f"[FALLBACK] {endpoint} | {level} | {message}")
        return False


def log_info(endpoint: str, message: str, **kwargs):
    """Raccourci pour log_to_db avec niveau INFO"""
    return log_to_db(endpoint, 'INFO', message, **kwargs)


def log_warning(endpoint: str, message: str, **kwargs):
    """Raccourci pour log_to_db avec niveau WARNING"""
    return log_to_db(endpoint, 'WARNING', message, **kwargs)


def log_error(endpoint: str, message: str, **kwargs):
    """Raccourci pour log_to_db avec niveau ERROR"""
    return log_to_db(endpoint, 'ERROR', message, **kwargs)


def log_debug(endpoint: str, message: str, **kwargs):
    """Raccourci pour log_to_db avec niveau DEBUG"""
    return log_to_db(endpoint, 'DEBUG', message, **kwargs)


def close_db_pool():
    """Ferme proprement le pool de connexions"""
    global _db_pool
    if _db_pool is not None:
        _db_pool.close()
        _db_pool = None
        logger.info("📊 Pool de connexions ingestion_logs fermé")
