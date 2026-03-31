"""
Analyze the class distribution of a Gold rome_dataset.parquet.

Prints a statistical summary of rome_code frequencies and helps identify:
- The shape of the distribution (skew, percentiles)
- How many classes would survive a given MIN_CLASS_COUNT filter
- The right capping threshold to balance the dataset for training

Usage:
    # Auto mode — latest Gold dataset dt
    python -m src.data.analyze_dataset

    # Explicit dt
    python -m src.data.analyze_dataset --dt 2026-02-13

    # Compare all dt partitions available in Gold
    python -m src.data.analyze_dataset --mode compare
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src.config.env import load_project_env
from src.storage.storage import get_storage_from_env

load_project_env()

logger = logging.getLogger(__name__)

DATASET_KEY_TEMPLATE = "datasets/{dt}/rome_dataset.parquet"


def resolve_dataset_dt(dt_arg: str | None) -> str:
    """Return the dt to load: explicit value or latest available in Gold datasets."""
    if dt_arg and dt_arg not in ("", "latest"):
        return dt_arg
    storage_gold = get_storage_from_env("gold")
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


def analyze(dt: str | None = None) -> dict:
    """
    Load the Gold dataset for the given dt and print a class distribution analysis.

    Args:
        dt: Dataset partition date (YYYY-MM-DD). None triggers auto mode.

    Returns:
        Dict with summary statistics and cap recommendations.
    """
    dt = resolve_dataset_dt(dt)
    dataset_key = DATASET_KEY_TEMPLATE.format(dt=dt)

    storage_gold = get_storage_from_env("gold")
    logger.info("Reading dataset: %s", dataset_key)
    df = storage_gold.read_parquet(dataset_key)

    counts = df["rome_code"].value_counts().sort_values(ascending=False)
    n_classes = len(counts)
    n_rows = len(df)

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_key}")
    print(f"{'='*60}")
    print(f"Total rows      : {n_rows:>10,}")
    print(f"Total classes   : {n_classes:>10,}")
    print(f"Avg per class   : {n_rows / n_classes:>10.0f}")

    print("\n--- Distribution statistics ---")
    print(f"Min             : {counts.min():>10,}")
    print(f"p10             : {int(counts.quantile(0.10)):>10,}")
    print(f"p25             : {int(counts.quantile(0.25)):>10,}")
    print(f"Median (p50)    : {int(counts.quantile(0.50)):>10,}")
    print(f"p75             : {int(counts.quantile(0.75)):>10,}")
    print(f"p90             : {int(counts.quantile(0.90)):>10,}")
    print(f"p95             : {int(counts.quantile(0.95)):>10,}")
    print(f"Max             : {counts.max():>10,}")

    p75 = int(counts.quantile(0.75))
    p90 = int(counts.quantile(0.90))

    # ------------------------------------------------------------------
    print("\n--- Impact MIN_CLASS_COUNT filter ---")
    print("  Interpretation : classes below the threshold are removed entirely.")
    print(f"  The current .env value is MIN_CLASS_COUNT={os.getenv('MIN_CLASS_COUNT', '25')}.")
    print()
    print(f"  {'Threshold':>10}  {'Classes kept':>14}  {'% classes':>10}  {'Rows kept':>10}  {'% rows':>8}  {'Rows dropped':>12}")
    print(f"  {'-'*72}")
    for threshold in [10, 25, 50, 100, 200]:
        surviving = (counts >= threshold).sum()
        dropped_classes = n_classes - surviving
        surviving_rows = int(counts[counts >= threshold].sum())
        dropped_rows = n_rows - surviving_rows
        print(
            f"  {f'>= {threshold}':>10}  "
            f"{surviving:>8} classes  "
            f"{surviving/n_classes*100:>8.1f}%  "
            f"{surviving_rows:>10,}  "
            f"{surviving_rows/n_rows*100:>7.1f}%  "
            f"{dropped_rows:>10,} ({dropped_classes} classes removed)"
        )

    # ------------------------------------------------------------------
    print("\n--- Impact capping (MAX_CLASS_COUNT — max examples per class) ---")
    print("  Interpretation : classes above the cap are randomly downsampled to N examples.")
    print(f"  Classes below the cap are untouched. p75={p75:,}  p90={p90:,}")
    print()
    print(f"  {'Cap':>7}  {'Classes affected':>17}  {'Rows kept':>10}  {'% kept':>7}  {'Rows removed':>13}  Note")
    print(f"  {'-'*78}")
    for cap in [200, 500, 1_000, 2_000, 5_000]:
        affected = (counts > cap).sum()
        capped_rows = int(counts.clip(upper=cap).sum())
        removed_rows = n_rows - capped_rows
        note = ""
        if cap <= p75:
            note = f"← aggressive: affects {affected/n_classes*100:.0f}% of classes (below p75)"
        elif cap <= p90:
            note = f"← moderate: affects top {affected/n_classes*100:.0f}% of classes (p75–p90)"
        else:
            note = f"← light: affects top {affected/n_classes*100:.0f}% of classes (above p90)"
        print(
            f"  {cap:>7,}  "
            f"{affected:>6} classes ({affected/n_classes*100:>4.1f}%)  "
            f"{capped_rows:>10,}  "
            f"{capped_rows/n_rows*100:>6.1f}%  "
            f"{removed_rows:>10,} rows  {note}"
        )

    # ------------------------------------------------------------------
    print("\n--- Top 10 most frequent classes ---")
    print("  Each class share = examples in class / total rows")
    print()
    for code, cnt in counts.head(10).items():
        bar = "█" * min(40, int(cnt / counts.max() * 40))
        share = cnt / n_rows * 100
        print(f"  {code} : {cnt:>6,}  ({share:>4.1f}% of dataset)  {bar}")

    print("\n--- Top 10 least frequent classes ---")
    print("  These classes are the most at risk of low recall without class_weight.")
    print()
    for code, cnt in counts.tail(10).items():
        share = cnt / n_rows * 100
        print(f"  {code} : {cnt:>6,}  ({share:>5.2f}% of dataset)")

    # ------------------------------------------------------------------
    print("\n--- Classes by frequency quintile ---")
    print(
        "  The dataset is split into 5 equal groups of classes ranked by frequency.\n"
        "  Q1 = rarest 20% of classes, Q5 = most frequent 20% of classes.\n"
        "  A healthy dataset has rows distributed roughly evenly across quintiles.\n"
        "  Heavy concentration in Q5 means LinearSVC will be biased toward those classes\n"
        "  and Q1 classes will have low recall — macro F1 will be degraded.\n"
    )
    labels = ["Q1 (rare)", "Q2", "Q3", "Q4", "Q5 (frequent)"]
    quintiles = pd.qcut(counts, q=5, labels=labels)
    print(f"  {'Quintile':<16}  {'Classes':>7}  {'Range':>16}  {'Rows':>8}  {'% of total':>10}  {'Avg/class':>10}")
    print(f"  {'-'*75}")
    for label in labels:
        group = counts[quintiles == label]
        group_rows = int(group.sum())
        print(
            f"  {label:<16}  {len(group):>7}  "
            f"[{group.min():>5,} – {group.max():>6,}]  "
            f"{group_rows:>8,}  "
            f"{group_rows/n_rows*100:>9.1f}%  "
            f"{group_rows/len(group):>9.0f}"
        )

    q5_rows = int(counts[quintiles == "Q5 (frequent)"].sum())
    q1_rows = int(counts[quintiles == "Q1 (rare)"].sum())
    print(
        f"\n  Q5 represents {q5_rows/n_rows*100:.1f}% of total rows vs "
        f"{q1_rows/n_rows*100:.1f}% for Q1 — imbalance ratio: {q5_rows/q1_rows:.0f}:1"
    )

    # ------------------------------------------------------------------
    # Compute concrete figures for each cap scenario
    classes_above_p75 = int((counts > p75).sum())
    classes_above_p90 = int((counts > p90).sum())
    rows_at_p75 = int(counts.clip(upper=p75).sum())
    rows_at_p90 = int(counts.clip(upper=p90).sum())
    speedup_p75 = n_rows / rows_at_p75
    speedup_p90 = n_rows / rows_at_p90

    print(f"\n{'='*60}")
    print("CONCLUSION — Recommended capping strategy")
    print(f"{'='*60}")
    print(
        "\n"
        f"  Current dataset : {n_rows:,} rows | {n_classes} classes | imbalance Q5/Q1 = {q5_rows/q1_rows:.0f}:1\n"
        "\n"
        f"  Option A — cap = p90 ({p90:,} examples)   [recommended for production]\n"
        f"  ┌─────────────────────────────────────────────────────┐\n"
        f"  │  Classes untouched : {n_classes - classes_above_p90:>4} / {n_classes}  ({(n_classes-classes_above_p90)/n_classes*100:.1f}%)            │\n"
        f"  │  Classes capped    : {classes_above_p90:>4} / {n_classes}  ({classes_above_p90/n_classes*100:.1f}%)            │\n"
        f"  │  Rows after cap    : {rows_at_p90:>10,}  ({rows_at_p90/n_rows*100:.1f}% of original)   │\n"
        f"  │  Rows removed      : {n_rows - rows_at_p90:>10,}  ({(n_rows-rows_at_p90)/n_rows*100:.1f}% reduction)       │\n"
        f"  │  Estimated speedup : ~{speedup_p90:.1f}x faster training               │\n"
        "  │  Risk              : minimal — only top 10% classes affected  │\n"
        "  └─────────────────────────────────────────────────────┘\n"
        "\n"
        f"  Option B — cap = p75 ({p75:,} examples)   [for fast iteration / testing]\n"
        "  ┌─────────────────────────────────────────────────────┐\n"
        f"  │  Classes untouched : {n_classes - classes_above_p75:>4} / {n_classes}  ({(n_classes-classes_above_p75)/n_classes*100:.1f}%)            │\n"
        f"  │  Classes capped    : {classes_above_p75:>4} / {n_classes}  ({classes_above_p75/n_classes*100:.1f}%)            │\n"
        f"  │  Rows after cap    : {rows_at_p75:>10,}  ({rows_at_p75/n_rows*100:.1f}% of original)   │\n"
        f"  │  Rows removed      : {n_rows - rows_at_p75:>10,}  ({(n_rows-rows_at_p75)/n_rows*100:.1f}% reduction)       │\n"
        f"  │  Estimated speedup : ~{speedup_p75:.1f}x faster training               │\n"
        "  │  Risk              : moderate — top 25% classes lose diversity │\n"
        "  └─────────────────────────────────────────────────────┘\n"
        "\n"
        "  Combined with class_weight='balanced' (already active in LinearSVC):\n"
        "  → The model will weight rare classes (Q1) higher during training.\n"
        "  → Expected macro F1 improvement even if accuracy stays flat.\n"
        "\n"
        f"  To apply: set MAX_CLASS_COUNT={p90} in .env and re-run make_dataset.\n"
    )

    print(f"{'='*60}\n")

    return {
        "dt": dt,
        "n_rows": n_rows,
        "n_classes": n_classes,
        "min": int(counts.min()),
        "max": int(counts.max()),
        "median": int(counts.median()),
        "p75": int(counts.quantile(0.75)),
        "p90": int(counts.quantile(0.90)),
    }


def _load_counts(storage_gold, dt: str) -> pd.Series | None:
    """Load rome_code value_counts for a given dt. Returns None if file missing."""
    key = DATASET_KEY_TEMPLATE.format(dt=dt)
    try:
        df = storage_gold.read_parquet(key)
        return df["rome_code"].value_counts()
    except Exception as e:
        logger.warning("Could not load dt=%s: %s", dt, e)
        return None


def compare() -> None:
    """
    Compare rome_code distributions across all dt partitions available in Gold datasets/.

    For each dt found under datasets/ in Gold, loads the dataset and computes:
    - Summary stats (rows, classes, median, p75, p90, Q5 share)
    - Classes that appeared or disappeared vs the previous dt
    - Top classes with the largest absolute frequency change between consecutive dt
    - Q1/Q5 row share evolution to detect worsening imbalance over time
    """
    storage_gold = get_storage_from_env("gold")
    prefixes = storage_gold.list_prefixes("datasets/")
    dts = sorted([p.strip("/") for p in prefixes if p.strip("/")])

    if not dts:
        raise RuntimeError("No dataset partitions found in Gold datasets/.")

    print(f"\n{'='*60}")
    print(f"MULTI-DT COMPARISON — {len(dts)} partition(s) found")
    print(f"Partitions: {', '.join(dts)}")
    print(f"{'='*60}")

    # Load all counts
    all_counts: dict[str, pd.Series] = {}
    for dt in dts:
        logger.info("Loading dt=%s ...", dt)
        counts = _load_counts(storage_gold, dt)
        if counts is not None:
            all_counts[dt] = counts

    if not all_counts:
        raise RuntimeError("No datasets could be loaded.")

    loaded_dts = sorted(all_counts.keys())

    # ------------------------------------------------------------------
    print("\n--- Summary per dt ---")
    print(
        f"  {'dt':<12}  {'Rows':>8}  {'Classes':>8}  {'Median':>7}  "
        f"{'p75':>6}  {'p90':>6}  {'Q5 share':>9}  {'Q1 share':>9}"
    )
    print(f"  {'-'*75}")
    for dt in loaded_dts:
        c = all_counts[dt]
        n_rows = int(c.sum())
        n_cls = len(c)
        labels = ["Q1 (rare)", "Q2", "Q3", "Q4", "Q5 (frequent)"]
        quintiles = pd.qcut(c, q=5, labels=labels)
        q5_share = c[quintiles == "Q5 (frequent)"].sum() / n_rows * 100
        q1_share = c[quintiles == "Q1 (rare)"].sum() / n_rows * 100
        print(
            f"  {dt:<12}  {n_rows:>8,}  {n_cls:>8,}  "
            f"{int(c.median()):>7,}  {int(c.quantile(0.75)):>6,}  "
            f"{int(c.quantile(0.90)):>6,}  {q5_share:>8.1f}%  {q1_share:>8.1f}%"
        )

    # ------------------------------------------------------------------
    print("\n--- Class stability between consecutive dt ---")
    print(
        "  'Appeared'  = class present in dt N but absent in dt N-1  (new ROME code in data)\n"
        "  'Disappeared' = class present in dt N-1 but absent in dt N  (code no longer in data)\n"
        "  A class that disappears means the model trained on dt N cannot predict it.\n"
    )
    for i in range(1, len(loaded_dts)):
        prev_dt, curr_dt = loaded_dts[i - 1], loaded_dts[i]
        prev_classes = set(all_counts[prev_dt].index)
        curr_classes = set(all_counts[curr_dt].index)
        appeared = sorted(curr_classes - prev_classes)
        disappeared = sorted(prev_classes - curr_classes)
        print(f"  {prev_dt} → {curr_dt}")
        print(f"    Appeared    : {len(appeared):>4} classes  {appeared[:10]}{'...' if len(appeared) > 10 else ''}")
        print(f"    Disappeared : {len(disappeared):>4} classes  {disappeared[:10]}{'...' if len(disappeared) > 10 else ''}")

    # ------------------------------------------------------------------
    print("\n--- Top 10 classes with largest frequency change between consecutive dt ---")
    print(
        "  Large swings indicate seasonal occupations or ingestion volume changes.\n"
        "  A class growing fast may dominate training; a shrinking class may lose recall.\n"
    )
    for i in range(1, len(loaded_dts)):
        prev_dt, curr_dt = loaded_dts[i - 1], loaded_dts[i]
        prev_c = all_counts[prev_dt]
        curr_c = all_counts[curr_dt]
        common = prev_c.index.intersection(curr_c.index)
        delta = (curr_c[common] - prev_c[common]).abs().sort_values(ascending=False)
        print(f"  {prev_dt} → {curr_dt}  (common classes: {len(common)})")
        print(f"  {'Code':<8}  {'Prev':>7}  {'Curr':>7}  {'Delta':>7}  {'Direction'}")
        print(f"  {'-'*45}")
        for code, diff in delta.head(10).items():
            prev_val = int(prev_c[code])
            curr_val = int(curr_c[code])
            direction = "▲" if curr_val > prev_val else "▼"
            print(f"  {code:<8}  {prev_val:>7,}  {curr_val:>7,}  {diff:>+7,}  {direction}")

    # ------------------------------------------------------------------
    print("\n--- Q5 share evolution (imbalance trend) ---")
    print(
        "  Q5 = most frequent 20%% of classes.\n"
        "  If Q5 share grows over time, the dataset becomes more skewed\n"
        "  and class_weight='balanced' becomes increasingly important.\n"
    )
    print(f"  {'dt':<12}  {'Q5 % of rows':>13}  {'Q1 % of rows':>13}  {'Imbalance ratio':>16}")
    print(f"  {'-'*58}")
    for dt in loaded_dts:
        c = all_counts[dt]
        n_rows = int(c.sum())
        labels = ["Q1 (rare)", "Q2", "Q3", "Q4", "Q5 (frequent)"]
        quintiles = pd.qcut(c, q=5, labels=labels)
        q5 = int(c[quintiles == "Q5 (frequent)"].sum())
        q1 = int(c[quintiles == "Q1 (rare)"].sum())
        ratio = q5 / q1 if q1 > 0 else float("inf")
        print(f"  {dt:<12}  {q5/n_rows*100:>12.1f}%  {q1/n_rows*100:>12.1f}%  {ratio:>14.0f}:1")

    print(f"\n{'='*60}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Analyze rome_code class distribution in Gold datasets."
    )
    parser.add_argument(
        "--dt",
        default=None,
        help="Partition date (YYYY-MM-DD). Omit for auto (latest Gold dt). Ignored in compare mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default="single",
        help="single: analyze one dt (default). compare: compare all available dt partitions.",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        compare()
    else:
        analyze(args.dt)


if __name__ == "__main__":
    main()
