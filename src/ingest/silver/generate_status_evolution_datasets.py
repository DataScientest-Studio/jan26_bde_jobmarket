"""
Build status evolution analytics datasets 

This script generates the analytic parquets (complete_dataset + status_timeline) from status_history and merged snapshots.

Trois modes sont disponibles :
- incremental : ne traite que le dernier snapshot (rapide, pour la prod quotidienne)
- recompute_all : reconstruit toute la timeline (pour validation ou correction massive)
- daily_reconstruction : (à venir) simule une exécution quotidienne pour chaque date de snapshot, en générant les sorties comme si le pipeline avait tourné chaque jour (audit, backfill, validation complète)


Usage :
- python -m src.ingest.silver.generate_status_evolution_datasets --mode incremental
- python -m src.ingest.silver.generate_status_evolution_datasets --mode recompute_all
- python -m src.ingest.silver.generate_status_evolution_datasets --mode daily_reconstruction

Main outputs (written under silver/status_analytics):
- dt={analysis_dt}/segment=complete_dataset/complete_offers_with_status.parquet
- dt={analysis_dt}/segment=status_timeline/status_timeline.parquet
- dt={analysis_dt}/run_id={job_id}/run.json

It provides a robust, auditable, and incremental dataset generation workflow.
"""


import argparse
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.storage.storage import Storage, get_storage_from_env
from src.utils import storage_tools
from src.utils.log_to_db import log_to_db

logger = logging.getLogger(__name__)

STATUS_FIELDS = [
	"status",
	"published_at",
	"unpublished_at",
	"reappeared_at",
	"first_seen_dt",
	"last_seen_dt",
]

def _extract_dt_from_key(key: str) -> Optional[str]:
	"""
	Extract snapshot date (dt=YYYY-MM-DD) from a storage key.
	Return the date string or None if not found.
	"""
	match = re.search(r"dt=(\d{4}-\d{2}-\d{2})", key)
	return match.group(1) if match else None

def _extract_merged_dt_from_key(key: str) -> Optional[str]:
	"""
	Extract merged snapshot date (merged_dt=YYYY-MM-DD) from a storage key.
	Return the date string or fallback to dt= if not found.
	"""
	merged_match = re.search(r"merged_dt=(\d{4}-\d{2}-\d{2})", key)
	if merged_match:
		return merged_match.group(1)
	return _extract_dt_from_key(key)

def _list_status_snapshot_files(storage_status: Storage) -> List[Tuple[str, str]]:
	"""
	List all status snapshot parquet files in storage, extract their snapshot dates from keys,
	and return sorted list of (snapshot_dt, key).

	Return format: List of tuples [(snapshot_dt, key), ...] sorted by snapshot_dt ascending.
	Only includes keys that match the expected pattern for status snapshots.

	Example :
		Input keys in storage_status:
			dt=2024-01-01/segment=offer_status/file1.parquet
			dt=2024-01-02/segment=offer_status/file2.parquet
		Output:
			[('2024-01-01', 'dt=2024-01-01/segment=offer_status/file1.parquet'),
			 ('2024-01-02', 'dt=2024-01-02/segment=offer_status/file2.parquet')]
	"""
	status_files = []
	for key in storage_status.list_keys(""):
		if not key.endswith(".parquet"):
			continue
		if "segment=offer_status/" not in key:
			continue
		dt = _extract_dt_from_key(key)
		if not dt:
			continue
		status_files.append((dt, key))
	status_files.sort(key=lambda x: x[0])
	return status_files

def _list_merged_files(storage_merged: Storage) -> List[Tuple[str, str]]:
	"""
	List all merged parquet files in storage_merged, extract their merged_dt from keys,
	and return sorted list of (merged_dt, key).

	Return format: List of tuples [(merged_dt, key), ...] sorted by merged_dt ascending.
	"""
	merged_files = []
	for key in storage_merged.list_keys(""):
		if not key.endswith(".parquet"):
			continue
		dt = _extract_merged_dt_from_key(key)
		if not dt:
			continue
		merged_files.append((dt, key))
	merged_files.sort(key=lambda x: x[0])
	return merged_files

def _build_complete_latest_dataset(
	latest_status_df: pd.DataFrame,
	latest_status_dt: str,
	latest_merged_df: pd.DataFrame,
	latest_merged_dt: str,
) -> pd.DataFrame:
	"""
	Build complete latest dataset by merging latest status snapshot with latest merged snapshot.

	This dataset represents the full state of all offers at the latest snapshot date, with status info.
	It is used as the reference dataset for the latest snapshot in the status timeline.

	Returns:
		- complete_df: DataFrame with one row per offer (id, source) at the latest snapshot date,
		- containing status fields from latest_status_df and merged fields from latest_merged_df.
	  The merge is done on (id, source) keys. If an offer is present in status but not in merged, 
	  it will still be included with NaNs for merged fields.
	"""
	required_keys = {"id", "source"}
	missing_status = required_keys.difference(latest_status_df.columns)
	missing_merged = required_keys.difference(latest_merged_df.columns)
	if missing_status:
		raise ValueError(f"latest status dataset missing required columns: {sorted(missing_status)}")
	if missing_merged:
		raise ValueError(f"latest merged dataset missing required columns: {sorted(missing_merged)}")
	merged_payload = latest_merged_df.copy()
	for c in STATUS_FIELDS:
		if c in merged_payload.columns:
			merged_payload = merged_payload.drop(columns=[c])
	complete_df = latest_status_df.merge(
		merged_payload,
		on=["id", "source"],
		how="left",
		indicator=True,
		suffixes=("", "_merged"),
	)
	complete_df["is_present_in_latest_merged"] = complete_df["_merge"].eq("both")
	complete_df = complete_df.drop(columns=["_merge"])
	complete_df["snapshot_dt"] = latest_status_dt
	complete_df["latest_merged_dt"] = latest_merged_dt
	return complete_df

def run_status_evolution_parquet_generation(
	mode: str = "incremental",
	storage_merged: Optional[Storage] = None,
	storage_status: Optional[Storage] = None,
	storage_analytics: Optional[Storage] = None,
	output_prefix: Optional[str] = None,
) -> Dict:
	"""
	Phase 1: generate dataset parquets only (complete_dataset + status_timeline).

	This function generates the raw data parquets without computing KPIs or charts.
	All analytical computations (KPIs, visualizations) are deferred to Phase 2.

	Modes:
	- incremental: only processes the latest snapshot (fast, for daily prod)
	- recompute_all: rebuilds the full timeline (for validation or massive correction)
	 """
	logger.info("=" * 80)
	logger.info(f"STATUS EVOLUTION ANALYTICS - PHASE 1: DATASET PARQUET GENERATION (mode={mode})")
	logger.info("=" * 80)
	job_id = f"status-analytics-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
	start_time = time.time()
	if storage_merged is None:
		storage_merged = get_storage_from_env("silver", "merged")
		logger.info("Storage (merged): silver/merged")
	if storage_status is None:
		storage_status = get_storage_from_env("silver", "status_history")
		logger.info("Storage (status_history): silver/status_history")
	if storage_analytics is None:
		storage_analytics = get_storage_from_env("silver", "status_analytics")
		logger.info("Storage (analytics): silver/status_analytics")
	try:
		log_to_db(
			endpoint="generate_status_evolution_datasets",
			level="INFO",
			message=f"Start status evolution dataset generation Phase 1: {job_id}",
			task_id=job_id,
			status="RUNNING",
		)
	except Exception as e:
		logger.warning("[generate_status_evolution_datasets Phase 1] log_to_db start failed: %s", e)

	# Load all status snapshot files and build timeline according to mode
	status_files = _list_status_snapshot_files(storage_status)
	if not status_files:
		raise RuntimeError("No status_history parquet files found")

	if mode == "recompute_all":
		# Timeline = concat of all snapshots (full history)
		timeline_parts = []
		for idx, (dt, key) in enumerate(status_files, start=1):
			logger.info(f"[{idx}/{len(status_files)}] Loading status snapshot {key}")
			df = storage_tools.load_parquet_dataset(storage_status, key)
			if df is None or len(df) == 0:
				continue
			df = df.copy()
			df["snapshot_dt"] = dt
			timeline_parts.append(df)
		if not timeline_parts:
			raise RuntimeError("Status snapshots exist but none could be loaded")
		timeline_df = pd.concat(timeline_parts, ignore_index=True, sort=False)
		latest_status_dt, latest_status_key = status_files[-1]
		latest_status_df = timeline_df[timeline_df["snapshot_dt"] == latest_status_dt].copy()
	else:
		# incremental: only load the latest snapshot
		latest_status_dt, latest_status_key = status_files[-1]
		logger.info(f"Loading latest status snapshot only: {latest_status_key}")
		latest_status_df = storage_tools.load_parquet_dataset(storage_status, latest_status_key)
		if latest_status_df is None or len(latest_status_df) == 0:
			raise RuntimeError(f"Failed to load latest status snapshot: {latest_status_key}")
		latest_status_df = latest_status_df.copy()
		latest_status_df["snapshot_dt"] = latest_status_dt
		timeline_df = latest_status_df.copy()

	# Always use the latest merged for the complete_dataset
	merged_files = _list_merged_files(storage_merged)
	if not merged_files:
		raise RuntimeError("No merged parquet files found")
	latest_merged_dt, latest_merged_key = merged_files[-1]
	logger.info(f"Loading latest merged dataset: {latest_merged_key}")
	latest_merged_df = storage_tools.load_parquet_dataset(storage_merged, latest_merged_key)
	if latest_merged_df is None or len(latest_merged_df) == 0:
		raise RuntimeError(f"Failed to load latest merged dataset: {latest_merged_key}")

	logger.info("Building complete latest-state dataset...")
	complete_latest_df = _build_complete_latest_dataset(
		latest_status_df=latest_status_df,
		latest_status_dt=latest_status_dt,
		latest_merged_df=latest_merged_df,
		latest_merged_dt=latest_merged_dt,
	)

	# Set Keys for saving outputs in analytics storage, partitioned by latest_status_dt
	analysis_dt = latest_status_dt
	complete_key = f"dt={analysis_dt}/segment=complete_dataset/complete_offers_with_status.parquet"
	timeline_key = f"dt={analysis_dt}/segment=status_timeline/status_timeline.parquet"
	if output_prefix:
		complete_key = f"dt={analysis_dt}/segment=complete_dataset/{output_prefix}_complete.parquet"
		timeline_key = f"dt={analysis_dt}/segment=status_timeline/{output_prefix}_timeline.parquet"

	# Save generated datasets as parquet in analytics storage
	logger.info("Saving dataset parquet outputs...")
	ok_complete = storage_tools.save_parquet_dataset(storage_analytics, complete_latest_df, complete_key)
	ok_timeline = storage_tools.save_parquet_dataset(storage_analytics, timeline_df, timeline_key)

	elapsed = time.time() - start_time
	success = ok_complete and ok_timeline

	# Generate a run.json metadata file.
	run_key = f"dt={analysis_dt}/run_id={job_id}/run.json"
	run_payload = {
		"job_id": job_id,
		"analysis_dt": analysis_dt,
		"timestamp_start": datetime.fromtimestamp(start_time).isoformat(),
		"timestamp_end": datetime.now().isoformat(),
		"elapsed_sec": elapsed,
		"phase": "parquet_generation",
		"inputs": {
			"latest_status_key": latest_status_key,
			"latest_merged_key": latest_merged_key,
		},
		"outputs": {
			"complete_key": complete_key,
			"timeline_key": timeline_key,
		},
		"rows": {
			"complete_latest": int(len(complete_latest_df)),
			"timeline": int(len(timeline_df)),
		},
		"success": success,
	}
	try:
		storage_analytics.write_json(run_key, run_payload)
	except Exception as e:
		logger.warning("Failed to write analytics run.json: %s", e)

	if success:
		logger.info("Phase 1: Dataset parquet generation completed successfully")
		logger.info("Complete dataset rows: %s", f"{len(complete_latest_df):,}")
		logger.info("Timeline rows: %s", f"{len(timeline_df):,}")
		try:
			log_to_db(
				endpoint="generate_status_evolution_datasets",
				level="INFO",
				message=(
					f"Status analytics Phase 1 completed: complete={len(complete_latest_df):,}, "
					f"timeline={len(timeline_df):,}"
				),
				task_id=job_id,
				status="SUCCESS",
				duration_sec=elapsed,
				records_count=int(len(timeline_df)),
				output_key=complete_key,
			)
		except Exception as e:
			logger.warning("[generate_status_evolution_datasets Phase 1] log_to_db success failed: %s", e)
	else:
		logger.error("Phase 1: Analytics failed while saving dataset outputs")
		try:
			log_to_db(
				endpoint="generate_status_evolution_datasets",
				level="ERROR",
				message="Phase 1: Failed to save dataset outputs",
				task_id=job_id,
				status="ERROR",
				duration_sec=elapsed,
			)
		except Exception as e:
			logger.warning("[generate_status_evolution_datasets Phase 1] log_to_db error failed: %s", e)

	return {
		"success": success,
		"job_id": job_id,
		"analysis_dt": analysis_dt,
		"complete_key": complete_key,
		"timeline_key": timeline_key,
		"phase": "parquet_generation",
		"run_key": run_key,
		"rows_complete": int(len(complete_latest_df)),
		"rows_timeline": int(len(timeline_df)),
		"elapsed_sec": elapsed,
	}

def run_daily_reconstruction():
    """
    Simule une exécution quotidienne pour chaque date de snapshot :
    pour chaque date, génère les sorties (complete_dataset, status_timeline, run.json)
    comme si le pipeline avait tourné chaque jour.
    """
    logger.info("Mode daily_reconstruction sélectionné : simulation d'une exécution quotidienne pour chaque date de snapshot.")
    storage_merged = get_storage_from_env("silver", "merged")
    storage_status = get_storage_from_env("silver", "status_history")
    storage_analytics = get_storage_from_env("silver", "status_analytics")

    status_files = _list_status_snapshot_files(storage_status)
    merged_files = _list_merged_files(storage_merged)
    if not status_files or not merged_files:
        logger.error("Aucun fichier status_history ou merged trouvé.")
        return

    for idx, (dt, status_key) in enumerate(status_files, start=1):
        logger.info(f"[daily_reconstruction] ({idx}/{len(status_files)}) Traitement snapshot {dt} : {status_key}")
        status_df = storage_tools.load_parquet_dataset(storage_status, status_key)
        if status_df is None or len(status_df) == 0:
            logger.warning(f"[daily_reconstruction] Snapshot {status_key} vide ou illisible, skip.")
            continue
        status_df = status_df.copy()
        status_df["snapshot_dt"] = dt
        timeline_parts = []
        for tdt, tkey in status_files:
            if tdt > dt:
                break
            tdf = storage_tools.load_parquet_dataset(storage_status, tkey)
            if tdf is not None and len(tdf) > 0:
                tdf = tdf.copy()
                tdf["snapshot_dt"] = tdt
                timeline_parts.append(tdf)
        if not timeline_parts:
            logger.warning(f"[daily_reconstruction] Timeline vide pour {dt}, skip.")
            continue
        timeline_df = pd.concat(timeline_parts, ignore_index=True, sort=False)
        merged_candidates = [(mdt, mkey) for mdt, mkey in merged_files if mdt <= dt]
        if not merged_candidates:
            logger.warning(f"[daily_reconstruction] Aucun merged parquet <= {dt}, skip.")
            continue
        latest_merged_dt, latest_merged_key = merged_candidates[-1]
        merged_df = storage_tools.load_parquet_dataset(storage_merged, latest_merged_key)
        if merged_df is None or len(merged_df) == 0:
            logger.warning(f"[daily_reconstruction] Merged {latest_merged_key} vide ou illisible, skip.")
            continue
        complete_latest_df = _build_complete_latest_dataset(
            latest_status_df=status_df,
            latest_status_dt=dt,
            latest_merged_df=merged_df,
            latest_merged_dt=latest_merged_dt,
        )
        analysis_dt = dt
        complete_key = f"dt={analysis_dt}/segment=complete_dataset/complete_offers_with_status.parquet"
        timeline_key = f"dt={analysis_dt}/segment=status_timeline/status_timeline.parquet"
        logger.info(f"[daily_reconstruction] Sauvegarde complete_dataset et status_timeline pour {dt}")
        ok_complete = storage_tools.save_parquet_dataset(storage_analytics, complete_latest_df, complete_key)
        ok_timeline = storage_tools.save_parquet_dataset(storage_analytics, timeline_df, timeline_key)
        job_id = f"daily-reconstruction-{dt}-{uuid.uuid4().hex[:8]}"
        run_key = f"dt={analysis_dt}/run_id={job_id}/run.json"
        run_payload = {
            "job_id": job_id,
            "analysis_dt": analysis_dt,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": datetime.now().isoformat(),
            "elapsed_sec": 0,
            "phase": "daily_reconstruction",
            "inputs": {
                "latest_status_key": status_key,
                "latest_merged_key": latest_merged_key,
            },
            "outputs": {
                "complete_key": complete_key,
                "timeline_key": timeline_key,
            },
            "rows": {
                "complete_latest": int(len(complete_latest_df)),
                "timeline": int(len(timeline_df)),
            },
            "success": bool(ok_complete and ok_timeline),
        }
        try:
            storage_analytics.write_json(run_key, run_payload)
        except Exception as e:
            logger.warning(f"[daily_reconstruction] Failed to write run.json for {dt}: {e}")
        logger.info(f"[daily_reconstruction] Terminé pour {dt} : complete={len(complete_latest_df):,}, timeline={len(timeline_df):,}")
    logger.info("daily_reconstruction terminé pour toutes les dates.")

def main() -> None:
	parser = argparse.ArgumentParser(description="Génération des datasets analytiques d'évolution de statut")
	parser.add_argument(
		"--mode",
		default="incremental",
		choices=["incremental", "recompute_all", "daily_reconstruction"],
		help="Mode d'exécution : incremental (dernier snapshot) ou recompute_all (toute la timeline)",
	)
	parser.add_argument(
		"--output-prefix",
		default=None,
		help="Préfixe optionnel pour les fichiers de sortie (pour éviter d'écraser les runs précédents)",
	)
	args = parser.parse_args()
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)
	if args.mode == "daily_reconstruction":
		run_daily_reconstruction()
		raise SystemExit(0)
	result = run_status_evolution_parquet_generation(mode=args.mode, output_prefix=args.output_prefix)

	if not result.get("success"):
		logger.error("Phase failed: %s", result)
		raise SystemExit(1)
	logger.info("Phase completed successfully.")
	raise SystemExit(0)

if __name__ == "__main__":
	main()
