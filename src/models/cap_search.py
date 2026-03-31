"""
Grid search over MAX_CLASS_COUNT values to find the optimal capping threshold.

For each cap value, runs a full training cycle (no tuning, update_latest=False)
and records metrics. Produces a comparison table and writes results to Gold.

The cap value controls the maximum number of examples per ROME class kept
before training. Lower cap = smaller dataset = faster training but potential
loss of lexical diversity on majority classes.

Key indicators to watch:
    - test/f1_macro : primary metric — improves when rare classes are better covered
    - test/accuracy : secondary — may drop slightly as majority classes are capped
    - training_duration_seconds : cost of each configuration
    - rows : dataset size after capping

Usage:
    # Default caps: 0 (no cap), p75, p90, and intermediate values
    python -m src.models.cap_search

    # Custom cap list
    python -m src.models.cap_search --caps 0,480,800,1561,3000

    # Specific dataset dt
    python -m src.models.cap_search --dt 2026-02-13 --caps 0,480,1561
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

from src.config.env import load_project_env
from src.storage.storage import get_storage_from_env
from src.models.train_model import train

load_project_env()
logger = logging.getLogger(__name__)

# Default cap values to test — covers: no cap, p75, intermediate, p90, above p90
DEFAULT_CAPS = [0, 480, 800, 1561, 2500, 5000]

# Results are written to Gold under this key
RESULTS_KEY_TEMPLATE = "models/cap_search/{dt}/results.json"


def run_cap_search(dt: str | None, caps: list[int]) -> list[dict]:
    """Run training for each cap value and return collected results."""
    results = []
    for cap in caps:
        cap_label = f"cap{cap}" if cap > 0 else "nocap"
        logger.info("--- cap_search: %s ---", cap_label)
        try:
            result = train(dt=dt, update_latest=False, max_class_count=cap if cap > 0 else 0)
            results.append({
                "cap": cap,
                "cap_label": cap_label,
                "dt": result["dt"],
                "rows": result["rows"],
                "classes": result["classes"],
                "effective_version": result["effective_version"],
                "training_duration_seconds": result["training_duration_seconds"],
                "metrics": result["metrics"],
            })
        except Exception as e:
            logger.error("cap=%s failed: %s", cap, e)
            results.append({"cap": cap, "cap_label": cap_label, "error": str(e)})
    return results


def print_summary(results: list[dict]) -> None:
    """Print a comparison table of all cap results."""
    header = (
        f"{'Cap':>8} | {'Rows':>7} | {'Classes':>7} | "
        f"{'Acc(test)':>9} | {'F1m(test)':>9} | "
        f"{'Top3(test)':>10} | {'Duration(s)':>11}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("CAP SEARCH — RESULTS SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"{r['cap']:>8} | ERROR: {r['error']}")
            continue
        m = r["metrics"]["test"]
        print(
            f"{r['cap_label']:>8} | {r['rows']:>7} | {r['classes']:>7} | "
            f"{m['accuracy']:>9.4f} | {m['f1_macro']:>9.4f} | "
            f"{m['top3']:>10.4f} | {r['training_duration_seconds']:>11.1f}"
        )
    print(sep)

    # Highlight best macro F1
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["metrics"]["test"]["f1_macro"])
        print(f"\n→ Best macro F1 : cap={best['cap_label']}  "
              f"f1_macro={best['metrics']['test']['f1_macro']:.4f}  "
              f"accuracy={best['metrics']['test']['accuracy']:.4f}")
    print()


def save_results(results: list[dict], dt: str) -> None:
    """Write results JSON to Gold storage."""
    storage_gold = get_storage_from_env("gold")
    key = RESULTS_KEY_TEMPLATE.format(dt=dt)
    payload = {
        "run_ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "dataset_dt": dt,
        "caps_tested": [r["cap"] for r in results],
        "results": results,
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    storage_gold.write_bytes(key, data, content_type="application/json; charset=utf-8")
    logger.info("Results saved to Gold: %s", key)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Grid search over MAX_CLASS_COUNT to find the optimal capping threshold."
    )
    parser.add_argument(
        "--dt",
        default=None,
        help="Dataset partition date (YYYY-MM-DD). Omit for auto (latest Gold dataset dt).",
    )
    parser.add_argument(
        "--caps",
        default=None,
        help=(
            "Comma-separated list of cap values to test (0 = no cap). "
            f"Default: {','.join(str(c) for c in DEFAULT_CAPS)}"
        ),
    )
    args = parser.parse_args()

    caps = (
        [int(c.strip()) for c in args.caps.split(",")]
        if args.caps
        else DEFAULT_CAPS
    )

    logger.info("cap_search — caps to test: %s", caps)
    results = run_cap_search(dt=args.dt, caps=caps)

    print_summary(results)

    # Use dt from first successful result for storage key
    valid = [r for r in results if "error" not in r]
    if valid:
        save_results(results, dt=valid[0]["dt"])


if __name__ == "__main__":
    main()
