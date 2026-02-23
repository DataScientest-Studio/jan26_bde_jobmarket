from src import data
from src.ingest.clients.france_travail_client import FranceTravailClient
from src.storage.storage import get_storage_from_env, Storage
from typing import Any, Dict, Iterable, List, Tuple
import os
import logging

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
        
    data = client.get(
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
            storage = get_storage_from_env(
                os.getenv("FT_DATA_DIR", "data/france_travail"),
                os.getenv("S3_PREFIX_FT", "france_travail")
            )

        # Récupérer les données
        logger.info("Début de l'ingestion des codes ROME métiers")
        codes_rome_metiers_sorted = get_rome_metiers()
        
        # Préparer les enregistrements
        key = "bronze/rome/rome_metiers.jsonl"
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