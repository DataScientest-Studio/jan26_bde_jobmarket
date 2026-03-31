"""
Script pour comparer les résultats des différentes stratégies de tuning.
Usage: python -m src.models.compare_results
"""

import json
from pathlib import Path
from typing import Dict, Any
import os

from src.config.env import load_project_env
load_project_env()

from src.storage.storage import get_storage_from_env

def load_metrics(storage, version: str) -> Dict[str, Any]:
    """Load metrics.json for a given model version"""
    try:
        key = f"models/rome_tfidf/versions/{version}/metrics.json"
        data = storage.read_bytes(key).decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print(f"⚠️  Could not load metrics for {version}: {e}")
        return None

def load_config(storage, version: str) -> Dict[str, Any]:
    """Load config.json for a given model version"""
    try:
        key = f"models/rome_tfidf/versions/{version}/config.json"
        data = storage.read_bytes(key).decode("utf-8")
        return json.loads(data)
    except Exception as e:
        print(f"⚠️  Could not load config for {version}: {e}")
        return None

def main():
    print("="*70)
    print("📊 COMPARAISON DES STRATÉGIES DE TUNING")
    print("="*70)
    
    storage = get_storage_from_env("gold")
    
    # Versions à comparer (correspondant au script test_all_strategies.ps1)
    versions = [
        ("v1_no_tuning", "No Tuning"),
        ("v2_manual", "Manual Tuning"),
        ("v3_random", "Random Search"),
        ("v4_grid", "Grid Search"),
    ]
    
    results = []
    
    for version, name in versions:
        print(f"\n{'─'*70}")
        print(f"📁 {name} ({version})")
        print(f"{'─'*70}")
        
        metrics = load_metrics(storage, version)
        config = load_config(storage, version)
        
        if metrics is None or config is None:
            print("❌ Modèle non trouvé. Exécuter d'abord: test_all_strategies.ps1")
            continue
        
        # Extract info
        test_metrics = metrics.get("test", {})
        tuning_info = config.get("tuning", {})
        strategy = tuning_info.get("strategy", "unknown")
        
        print(f"Strategy: {strategy}")
        print("\n📈 Test Metrics:")
        print(f"  - Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
        print(f"  - F1-Macro:  {test_metrics.get('f1_macro', 0):.4f}")
        print(f"  - Top-3:     {test_metrics.get('top3', 0):.4f}")
        print(f"  - Top-5:     {test_metrics.get('top5', 0):.4f}")
        
        if strategy != "none":
            print("\n🔧 Best Parameters:")
            best_params = tuning_info.get("best_params", {})
            for key, value in best_params.items():
                print(f"  - {key}: {value}")
            
            if "best_cv_score" in tuning_info:
                print(f"\n📊 Cross-Validation Score: {tuning_info['best_cv_score']:.4f}")
            
            if "total_combinations_tested" in tuning_info:
                print(f"🔍 Combinations tested: {tuning_info['total_combinations_tested']}")
        
        # Store for comparison table
        results.append({
            "name": name,
            "version": version,
            "accuracy": test_metrics.get('accuracy', 0),
            "f1_macro": test_metrics.get('f1_macro', 0),
            "top3": test_metrics.get('top3', 0),
            "top5": test_metrics.get('top5', 0),
            "strategy": strategy,
        })
    
    # Summary table
    if results:
        print(f"\n{'='*70}")
        print("📊 TABLEAU COMPARATIF")
        print(f"{'='*70}")
        print(f"{'Strategy':<20} {'Accuracy':<10} {'F1-Macro':<10} {'Top-3':<10} {'Top-5':<10}")
        print(f"{'-'*70}")
        
        # Sort by accuracy
        results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        
        for i, r in enumerate(results_sorted):
            prefix = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{prefix} {r['name']:<18} {r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {r['top3']:<10.4f} {r['top5']:<10.4f}")
        
        print(f"{'='*70}")
        
        best = results_sorted[0]
        print(f"\n🏆 MEILLEURE STRATÉGIE: {best['name']}")
        print(f"   Test Accuracy: {best['accuracy']:.4f}")
        print(f"   Test F1-Macro: {best['f1_macro']:.4f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
