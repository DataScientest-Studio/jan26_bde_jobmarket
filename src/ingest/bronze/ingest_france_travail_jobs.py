""" 
==============
BRONZE Layer
==============
 Ingest script for France Travail (FT) job offers ingestion via API.

This script performs the following steps:

1. Get the list of ROME codes (job categories) from the France Travail API.
2. For each ROME code, query the total number of job offers available.
3. If the total is below a defined threshold (FT_MAX_RETRIEVABLE), retrieve all offers in a single pass.
4. If the total exceeds the threshold, split the retrieval into time windows (e.g., 7-day windows) to ensure we stay within API limits.
5. Store the retrieved offers in the bronze layer storage, partitioned by date and ROME code for efficient querying in later stages.

The main function `ingest_france_travail_offers` can be called from a CLI or scheduled to run periodically (e.g., daily)
It can be called via a microservice endpoint in FastAPI, allowing for on-demand ingestion or pipeline integration.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

# ----------------------------
# Load environment variables
# ----------------------------
from src.config.env import load_project_env

# ----------------------------
# Import project-specific modules
# ----------------------------
from src.ingest.clients.france_travail_client import FranceTravailClient
from src.ingest.tools.france_travail_common import get_rome_metiers, print_rome_line, probe_total, extract_and_store_by_range, extract_and_store_by_windows, write_run_metadata

from src.utils.time_helpers import utc_dt_str, utc_run_id, format_eta
from src.storage.storage import get_storage_from_env, Storage
from src.utils.log_to_db import log_to_db

load_project_env()  # safe à rappeler (idempotent)

# Logger setup
logger = logging.getLogger(__name__)
structured_logger = logging.getLogger("structured")

FT_MAX_RETRIEVABLE = int(os.getenv("FT_MAX_RETRIEVABLE", "3150"))


def ingest_france_travail_offers(
    storage: Storage = None,
    client: FranceTravailClient = None,
    window_days: int = None,
    max_windows: int = None,
    binary_split_min_seconds: int = None,
    max_rome_codes: int = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    logger_override: Optional[logging.Logger] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Service d'ingestion des offres d'emploi France Travail en couche bronze.
    
    Args:
        storage: Storage backend (optionnel, créé depuis env si non fourni)
        client: Client France Travail (optionnel, créé si non fourni)
        window_days: Taille des fenêtres temporelles en jours
        max_windows: Nombre maximum de fenêtres
        binary_split_min_seconds: Taille minimale de fenêtre pour split binaire
        max_rome_codes: Limiter le nombre de codes ROME traités (0 = tous)
        progress_callback: Callback appelé avec (current, total, rome_code, rome_label) à chaque progression
        logger_override: Logger personnalisé à utiliser (optionnel)
        
    Returns:
        Dict avec le statut de l'opération et les statistiques détaillées
    """
    # Utiliser le logger passé ou le logger du module
    log = logger_override if logger_override is not None else logger
    
    try:
        # Initialize storage and client
        if storage is None:
            storage = get_storage_from_env("bronze", "france_travail")
        
        if client is None:
            client = FranceTravailClient()
        
        # Get configuration with defaults
        window_days = window_days or int(os.getenv("FT_WINDOW_DAYS", "7"))
        max_windows = max_windows or int(os.getenv("FT_MAX_WINDOWS", "260"))
        binary_split_min_seconds = binary_split_min_seconds or int(os.getenv("FT_BINARY_SPLIT_MIN_SECONDS", "3600"))
        max_rome_codes = max_rome_codes if max_rome_codes is not None else int(os.getenv("FT_MAX_ROME_CODES", "0"))
        
        # Generate run metadata
        dt = utc_dt_str()
        run_id = utc_run_id()
        enable_grafana = os.getenv("ENABLE_GRAFANA_LOGS", "false").lower() == "true"

        def emit_structured(event_type: str, **fields: Any) -> None:
            if not enable_grafana:
                return
            payload: Dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "task_type": "france_travail_offers",
                "run_id": run_id,
            }
            if task_id:
                payload["task_id"] = task_id
            payload.update(fields)
            if "records_count" not in payload and "records_written" in payload:
                payload["records_count"] = payload["records_written"]
            structured_logger.info(json.dumps(payload))
        
        log.info(f"Début de l'ingestion France Travail - run_id={run_id}")
        
        # Base query parameters
        base_params: Dict[str, Any] = {"sort": "1"}
        
        # Fetch ROME codes
        log.info("Récupération de la liste des codes ROME...")
        rome_items = get_rome_metiers(client)
        
        if max_rome_codes > 0:
            rome_items = rome_items[:max_rome_codes]
            log.info(f"Limitation à {max_rome_codes} codes ROME")
        
        log.info(f"Traitement de {len(rome_items)} codes ROME")
        emit_structured(
            "ingestion_started",
            rome_total=len(rome_items),
            window_days=window_days,
            max_windows=max_windows,
            max_rome_codes=max_rome_codes,
        )
        
        # Process each ROME code
        total_errors = 0
        all_skipped_ranges: List[Dict[str, Any]] = []
        per_rome_stats: List[Dict[str, Any]] = []
        total_calls = 0
        total_written = 0
        started = time.time()
        last_rome_time = started
        
        for idx, rome in enumerate(rome_items, 1):
            log.info(f"[{idx}/{len(rome_items)}] Traitement {rome.code} - {rome.libelle}")
            
            # Callback de progression
            if progress_callback:
                progress_callback(idx, len(rome_items), rome.code, rome.libelle)
            
            total_global = probe_total(client, {"codeROME": rome.code, **base_params})
            emit_structured(
                "rome_progress",
                rome_code=rome.code,
                rome_label=rome.libelle,
                rome_index=idx,
                rome_total=len(rome_items),
                total_global=total_global,
            )
            print_rome_line(rome.code, total_global, rome.libelle)
            
            # Log progress to database for real-time monitoring
            current_time = time.time()
            elapsed_so_far = current_time - started
            rome_duration = current_time - last_rome_time
            pct = (idx / len(rome_items)) * 100 if len(rome_items) > 0 else 0.0
            rate = idx / elapsed_so_far if elapsed_so_far > 0 else 0.0
            remaining = len(rome_items) - idx
            eta_seconds = remaining / rate if rate > 0 else 0.0
            rome_task_id_progress = f"{run_id}-rome-{idx}"
            log_to_db(
                endpoint='france_travail_offers',
                level='INFO',
                message=(
                    f"⏳ Code ROME {idx}/{len(rome_items)} ({pct:.1f}%): {rome.code} - {rome.libelle} "
                    f"({total_global} offres) - Écoulé: {format_eta(elapsed_so_far)} - ETA: {format_eta(eta_seconds)}"
                ),
                task_id=rome_task_id_progress,
                duration_sec=round(rome_duration, 2),
                records_count=total_written,
                extra_metadata={
                    'rome_code': rome.code,
                    'rome_label': rome.libelle,
                    'rome_index': idx,
                    'rome_total': len(rome_items),
                    'percent': round(pct, 1),
                    'total_global': total_global,
                    'rome_duration': round(rome_duration, 2),
                    'elapsed_total': round(elapsed_so_far, 2),
                    'eta_seconds': round(eta_seconds, 1),
                    'run_id': run_id
                }
            )
            log.info(
                "Progress ROME %s/%s (%.1f%%) | code=%s | total=%s | elapsed=%s | ETA=%s",
                idx,
                len(rome_items),
                pct,
                rome.code,
                total_global,
                format_eta(elapsed_so_far),
                format_eta(eta_seconds),
            )
            last_rome_time = current_time
            
            if total_global <= FT_MAX_RETRIEVABLE:
                res = extract_and_store_by_range(
                    client=client,
                    storage=storage,
                    dt=dt,
                    run_id=run_id,
                    code_rome=rome.code,
                    segment="global",
                    base_params=base_params,
                    page_size=150,
                )
                
                stat = {
                    "code": rome.code,
                    "libelle": rome.libelle,
                    "total_global": total_global,
                    "mode": res["mode"],
                    "calls": res["calls"],
                    "written": res["written"],
                }
            else:
                res = extract_and_store_by_windows(
                    client=client,
                    storage=storage,
                    dt=dt,
                    run_id=run_id,
                    code_rome=rome.code,
                    base_params=base_params,
                    total_global=total_global,
                    window_days=window_days,
                    max_windows=max_windows,
                    binary_split_min_seconds=binary_split_min_seconds,
                )
                
                stat = {
                    "code": rome.code,
                    "libelle": rome.libelle,
                    "total_global": total_global,
                    "mode": res["mode"],
                    "window_days": res["window_days"],
                    "windows_used": res["windows_used"],
                    "sum_windows": res["sum_windows"],
                    "calls": res["calls"],
                    "written": res["written"],
                }
            
            total_errors += res.get("errors", 0)
            all_skipped_ranges.extend(res.get("skipped_ranges", []))
            per_rome_stats.append(stat)
            total_calls += stat["calls"]
            total_written += stat["written"]

            emit_structured(
                "rome_completed",
                rome_code=rome.code,
                rome_label=rome.libelle,
                rome_index=idx,
                rome_total=len(rome_items),
                total_global=total_global,
                mode=stat.get("mode"),
                calls=stat["calls"],
                records_written=stat["written"],
            )
        
        elapsed = time.time() - started
        
        # Prepare run payload
        run_payload = {
            "run_id": run_id,
            "dt": dt,
            "params": {
                "window_days": window_days,
                "max_windows": max_windows,
                "binary_split_min_seconds": binary_split_min_seconds,
                "max_rome_codes": max_rome_codes,
                "max_retrievable": FT_MAX_RETRIEVABLE,
            },
            "stats": {
                "rome_processed": len(per_rome_stats),
                "calls": total_calls,
                "written": total_written,
                "elapsed_s": round(elapsed, 2),
                "errors": total_errors,
                "skipped_ranges": len(all_skipped_ranges),
            },
            "storage": {
                "backend": os.getenv("STORAGE_BACKEND", "local"),
                "s3_bucket": os.getenv("S3_BUCKET", ""),
                "s3_prefix": os.getenv("S3_PREFIX_FT", ""),
                "local_root": os.getenv("FT_DATA_DIR", ""),
            },
            "errors": {
                "total_errors": total_errors,
                "skipped_ranges_count": len(all_skipped_ranges),
                "skipped_ranges": all_skipped_ranges,
            },
            "per_rome": per_rome_stats,
        }
        
        # Write run metadata
        run_key = write_run_metadata(storage, run_id, run_payload)
        
        log.info(f"Ingestion terminée: {total_written} offres écrites en {elapsed:.2f}s")
        throughput = round(total_written / elapsed, 2) if elapsed > 0 else 0
        emit_structured(
            "ingestion_summary",
            rome_processed=len(per_rome_stats),
            records_count=total_written,
            records_written=total_written,
            duration_sec=round(elapsed, 2),
            throughput=throughput,
            calls=total_calls,
            errors=total_errors,
        )
        
        return {
            "success": True,
            "run_id": run_id,
            "dt": dt,
            "run_key": run_key,
            "rome_processed": len(per_rome_stats),
            "calls": total_calls,
            "written": total_written,
            "elapsed_s": round(elapsed, 2),
            "errors": total_errors,
            "skipped_ranges": len(all_skipped_ranges),
            "message": f"Ingestion réussie: {total_written} offres écrites, {len(per_rome_stats)} codes ROME traités en {elapsed:.2f}s",
            "details": run_payload
        }
        
    except Exception as e:
        log.error(f"Erreur lors de l'ingestion France Travail: {e}", exc_info=True)
        if 'emit_structured' in locals():
            emit_structured("ingestion_error", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "message": f"Échec de l'ingestion: {e}"
        }


def main_cli() -> None:
    """Point d'entrée CLI pour l'ingestion des offres France Travail."""
    result = ingest_france_travail_offers()
    
    if result["success"]:
        print(f"\n✅ {result['message']}")
        print(f"run_id={result['run_id']}")
        print(f"run_key={result['run_key']}")
        print(f"calls={result['calls']}\twritten={result['written']}\telapsed_s={result['elapsed_s']:.2f}")
    else:
        print(f"\n❌ {result['message']}")
        exit(1)


if __name__ == "__main__":
    main_cli()

