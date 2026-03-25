"""
DAG expérimental — Grid search sur MAX_CLASS_COUNT.

Entraîne un modèle pour chaque valeur de cap, dans l'ordre défini par CAP_VALUES,
sans mettre à jour LATEST.json (mode expérience).

Les tâches s'exécutent séquentiellement pour éviter la contention mémoire
(chaque run SGDClassifier(hinge) + TF-IDF peut consommer 1-2 Go de RAM).
Le modèle utilise SGDClassifier(loss='hinge') à la place de LinearSVC :
même objectif hinge loss, mais mémoire O(n_features) vs O(n_samples×n_classes).
Voir docs/ML.md pour le détail.

Résultats sauvegardés dans Gold : models/cap_search/{dt}/results.json
Le tableau comparatif est visible dans les logs de la tâche save_results.

Déclenchement :
    - Manuel via "Trigger DAG" dans l'UI Airflow
    - Paramètres configurables au déclenchement : dt, caps

DAG non schedulé (schedule=None) — expérimental uniquement.

"""
import os
import sys

# Ajoute la racine du projet (/opt/airflow) au PYTHONPATH pour que `src` soit importable.
# Nécessaire car Airflow 3.x exécute les tâches dans des subprocesses qui n'héritent
# pas du PYTHONPATH défini dans docker-compose.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airflow import DAG
from airflow.sdk import Param
from airflow.providers.standard.operators.python import PythonOperator

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Ordre des caps : du plus restrictif au plus permissif, nocap en dernier
# pour avoir les résultats intermédiaires rapidement.
# Valeur à adapter selon la distribution des classes du dataset (voir src/data/analyze_dataset.py) :
# Ex 
# dt                Rows        Classes    Median  p75   p90       Q5 share   Q1 share
#    ---------------------------------------------------------------------------
#    2026-02-13     552,589       972      168     480   1,561      75.5%       1.4%
#    2026-02-16     551,783       969      166     478   1,546      75.6%       1.4%
#    2026-03-02     340,075       576      202     583   1,663      73.2%       1.5%
#    2026-03-03     581,943       982      175     501   1,609      76.2%       1.3%
#    2026-03-07     599,935       993      173     502   1,632      76.2%       1.3%
#    2026-03-09     596,852       992      172     500   1,628      76.3%       1.3%
DEFAULT_CAP_VALUES = [200, 400, 600, 800, 1000, 1500, 0]

# ---------------------------------------------------------------------------
# Tâches
# ---------------------------------------------------------------------------

def _train_cap(cap: int, **context) -> dict:
    """Entraîne un modèle avec le cap donné et pousse les métriques en XCom."""
    import os, sys
    # Le DAG est dans /opt/airflow/dags/ — la racine du projet est un niveau au-dessus.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.config.env import load_project_env
    load_project_env()
    from src.models.train_model import train

    dt = context["params"].get("dt") or None
    result = train(dt=dt, update_latest=False, max_class_count=cap if cap > 0 else 0)
    return result


def _save_and_print_results(**context) -> None:
    """Collecte les résultats XCom de toutes les tâches cap_* et affiche le tableau."""
    import json
    from datetime import timezone
    from src.config.env import load_project_env
    load_project_env()
    from src.storage.storage import get_storage_from_env

    ti = context["task_instance"]
    cap_values = context["params"].get("caps", DEFAULT_CAP_VALUES)

    results = []
    for cap in cap_values:
        task_id = f"cap_{cap if cap > 0 else 'nocap'}"
        result = ti.xcom_pull(task_ids=task_id)
        if result:
            results.append({
                "cap": cap,
                "cap_label": f"cap{cap}" if cap > 0 else "nocap",
                **result,
            })
        else:
            results.append({"cap": cap, "cap_label": f"cap{cap}" if cap > 0 else "nocap", "error": "no xcom result"})

    # --- Tableau comparatif ---
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
        if "error" in r and "metrics" not in r:
            print(f"{r['cap_label']:>8} | ERROR: {r.get('error')}")
            continue
        m = r["metrics"]["test"]
        print(
            f"{r['cap_label']:>8} | {r['rows']:>7} | {r['classes']:>7} | "
            f"{m['accuracy']:>9.4f} | {m['f1_macro']:>9.4f} | "
            f"{m['top3']:>10.4f} | {r['training_duration_seconds']:>11.1f}"
        )
    print(sep)

    valid = [r for r in results if "metrics" in r]
    if valid:
        best = max(valid, key=lambda r: r["metrics"]["test"]["f1_macro"])
        print(
            f"\n→ Best macro F1 : cap={best['cap_label']}  "
            f"f1_macro={best['metrics']['test']['f1_macro']:.4f}  "
            f"accuracy={best['metrics']['test']['accuracy']:.4f}"
        )
    print()

    # --- Sauvegarde Gold ---
    if not valid:
        print("Aucun résultat valide — rien à sauvegarder.")
        return

    dt = valid[0]["dt"]
    storage_gold = get_storage_from_env("gold")
    key = f"models/cap_search/{dt}/results.json"
    payload = {
        "run_ts_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "dataset_dt": dt,
        "caps_tested": cap_values,
        "results": results,
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    storage_gold.write_bytes(key, data, content_type="application/json; charset=utf-8")
    print(f"Résultats sauvegardés : {key}")


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="cap_search",
    description="Grid search expérimental sur MAX_CLASS_COUNT — non schedulé",
    schedule=None,                        # déclenchement manuel uniquement
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["jobmarket", "ml", "experiment"],
    doc_md="""
        DAG pour tester différents plafonds de MAX_CLASS_COUNT et leur impact sur les métriques du modèle. 
        La distribution des classes dans le dataset est très déséquilibrée, avec une longue traîne de classes rares.
        La distribution sur un dataset est analysée dans src/data/analyze_dataset.py, 
        sur l'ensemble des dt disponibles dans Gold (mode compare) les résultats sont :
        - p50 = 170 (50% des classes ont ≤170 exemples)
        - p75 = 500 (75% des classes ont ≤500 exemples)
        - p90 autour de 1600 (90% des classes ont ≤1600 exemples)

        """,    
    params={
        "dt": Param(
            None,
            type=["null", "string"],
            description="Dataset dt à utiliser (YYYY-MM-DD). Vide = dernier dt disponible.",
        ),
        "caps": Param(
            DEFAULT_CAP_VALUES,
            type="array",
            description="Liste des caps à tester (0 = pas de cap).",
        ),
    },
    default_args={
        "owner": "jobmarket",
        "retries": 0,
        "execution_timeout": timedelta(hours=4),
    },
) as dag:

    cap_tasks = []
    for cap in DEFAULT_CAP_VALUES:
        task_id = f"cap_{cap if cap > 0 else 'nocap'}"
        task = PythonOperator(
            task_id=task_id,
            python_callable=_train_cap,
            op_kwargs={"cap": cap},
        )
        cap_tasks.append(task)

    save_results = PythonOperator(
        task_id="save_results",
        python_callable=_save_and_print_results,
    )

    # Exécution séquentielle : cap_500 → cap_1000 → cap_1628 → cap_2500 → cap_nocap → save_results
    for i in range(len(cap_tasks) - 1):
        cap_tasks[i] >> cap_tasks[i + 1]
    cap_tasks[-1] >> save_results
