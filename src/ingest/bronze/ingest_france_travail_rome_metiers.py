""" 
==============
BRONZE Layer
==============
 Ingest France Travail (FT) ROME referential data into the bronze layer.
 
 ROME is the French national classification of jobs, with codes and labels for each métier (job category).
 This script retrieves the list of ROME codes and their corresponding labels from the France Travail API,
 and stores them in the bronze layer storage as a JSONL file for later use in the data pipeline.

The main function `ingest_rome_metiers` can be called from a CLI or scheduled to run periodically (e.g., daily)
to keep the ROME referential data up to date. It handles errors gracefully and logs the process for observability.

This import function is also exposed in FastAPI to pilot ingestion via a micoservice endpoint, 
allowing for on-demand ingestion or integration with other systems.
 
"""

from src.ingest.clients.france_travail_client import FranceTravailClient
from src.storage.storage import get_storage_from_env, Storage
from typing import Any, Dict, List, Tuple
import logging

from src.config.env import require_env, get_project_root, load_project_env

load_project_env()  # safe à rappeler (idempotent)
logger = logging.getLogger(__name__)


def get_rome_metiers(client: FranceTravailClient = None) -> List[Tuple[str, str]]:
    """
    Récupère les codes ROME métiers depuis l'API France Travail.
    
    Args:
        client: Client France Travail (optionnel, créé si non fourni)
        
    Returns:
        Liste triée de tuples (code, libelle)
    """
    if client is None:
        client = FranceTravailClient()
        
        client.get(
        "https://api.francetravail.io/partenaire/rome-metiers/v1/metiers/metier",
        params={
            "champs": "code,appellations(libelle)"
        },
    )

    codes_rome_metiers = [(item["code"], item["appellations"][0]["libelle"]) for item in data]
    codes_rome_metiers_sorted = sorted(codes_rome_metiers)
    return codes_rome_metiers_sorted


def ingest_rome_metiers(storage: Storage = None) -> Dict[str, Any]:
    """
    Service d'ingestion des codes ROME métiers.
    Récupère les données et les écrit dans le storage.
    
    Args:
        storage: Storage backend (optionnel, créé depuis env si non fourni)
        
    Returns:
        Dict avec le statut de l'opération et les statistiques
    """
    try:
        # Initialiser le storage si non fourni
        if storage is None:
            storage = get_storage_from_env("bronze", "france_travail")

        # Récupérer les données
        logger.info("Début de l'ingestion des codes ROME métiers")
        codes_rome_metiers_sorted = get_rome_metiers()
        
        # Préparer les enregistrements
        key = "rome/rome_metiers.jsonl"
        records = [{"code": code, "libelle": libelle} for code, libelle in codes_rome_metiers_sorted]
        
        # Écrire dans le storage
        written = storage.write_jsonl(key, records)
        
        result = {
            "success": True,
            "key": key,
            "records_count": len(codes_rome_metiers_sorted),
            "records_written": written,
            "message": f"Ingestion réussie: {written} codes ROME métiers écrits dans {key}"
        }
        
        logger.info(result["message"])
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'ingestion des codes ROME: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": f"Échec de l'ingestion: {e}"
        }


def main():
    """Point d'entrée CLI pour l'ingestion des codes ROME métiers."""
    result = ingest_rome_metiers()
    
    if result["success"]:
        print(f"✅ {result['message']}")
        print(f"Nombre de métiers dans la nomenclature ROME: {result['records_count']}")
    else:
        print(f"❌ {result['message']}")
        exit(1)


if __name__ == "__main__":
    main()