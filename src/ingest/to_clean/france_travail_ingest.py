from src.ingest.france_travail_client import FranceTravailClient

def main():

    client = FranceTravailClient()

    # Exemple : recherche paginée
    data = client.get(
        "/partenaire/offresdemploi/v2/offres/search",
        params={
#            "motsCles": "data engineer",
#            "departement": "75",
            "range": "1-5",
#            "sort": "1",
        },
    )

    print(len(data.get("resultats", [])))    

if __name__ == "__main__":
    main()