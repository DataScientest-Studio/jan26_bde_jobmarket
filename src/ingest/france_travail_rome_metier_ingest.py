from src import data
from src.ingest.france_travail_client import FranceTravailClient
from src.storage.storage import get_storage_from_env, Storage
from typing import Any, Dict, Iterable, List, Tuple
import os


def get_rome_metiers():

    client = FranceTravailClient()
    data = client.get(
        "https://api.francetravail.io/partenaire/rome-metiers/v1/metiers/metier",
        params={
            "champs" : "code,appellations(libelle)"
        },
    )

    codes_rome_metiers = [(item["code"], item["appellations"][0]["libelle"]) for item in data]
    codes_rome_metiers_sorted = sorted(codes_rome_metiers)
    return codes_rome_metiers_sorted

def main():

    storage = get_storage_from_env(os.getenv("FT_DATA_DIR", "data/france_travail"),
                                   os.getenv("S3_PREFIX_FT", "france_travail"))

    codes_rome_metiers_sorted =get_rome_metiers()
    key="bronze/rome/rome_metiers.jsonl"
      # Conversion en liste de dicts pour JSONL
    records = [ {"code": code, "libelle": libelle} for code, libelle in codes_rome_metiers_sorted]
    written = storage.write_jsonl(key, records)
    print(f"Écriture de {len(codes_rome_metiers_sorted)} codes ROME métiers dans {key} : {'succès' if written else 'échec'}")

    print("Nombre de métiers dans la nomenclature ROME : ", len(codes_rome_metiers_sorted)) 

if __name__ == "__main__":
    main()