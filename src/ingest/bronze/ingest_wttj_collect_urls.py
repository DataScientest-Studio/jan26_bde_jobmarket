"""
Welcome to the Jungle - Collecteur d'URLs via le Sitemap XML
=============================================================
TÃ©lÃ©charge et parse le sitemap WTTJ pour extraire les URLs d'offres.
Approche 100% publique, sans restriction d'API ni de Referer.

Le rÃ©sultat est un fichier urls.txt prêt à  être utilisé avec wttj_scraper.py

Usage:
    # Toutes les offres FR (jusqu'Ã  1000 par dÃ©faut)
    python wttj_collect_urls.py

    # Filtrer par mot-clÃ© dans l'URL (intituléde poste)
    python wttj_collect_urls.py --query "manager"
    python wttj_collect_urls.py --query "data-engineer"

    # Filtrer par slug d'entreprise
    python wttj_collect_urls.py --entreprise "auchan"
    python wttj_collect_urls.py --entreprise "decathlon"

    # Filtrer par ville (dans l'URL)
    python wttj_collect_urls.py --ville "paris"
    python wttj_collect_urls.py --ville "lyon"

    # Limiter le nombre de rÃ©sultats
    python wttj_collect_urls.py --max 500

    # Tout combiner
    python wttj_collect_urls.py --query "manager" --ville "paris" --max 200

    # Changer le fichier de sortie
    python wttj_collect_urls.py --output mes_urls.txt

Installation:
    pip install requests
"""

import argparse
import gzip
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

#  Load project environment
# ─────────────────────────────────────────────
#  Import project-specific modules
# ─────────────────────────────────────────────
from src.utils.time_helpers import utc_run_id, utc_now_iso
from src.storage.storage import get_storage_from_env

# ─────────────────────────────────────────────
#  Load environment variables
# ─────────────────────────────────────────────
from src.config.env import load_project_env
load_project_env()  # safe à rappeler (idempotent)


#  Config

BASE_URL      = os.getenv("WTTJ_BASE_URL", "https://www.welcometothejungle.com")
SITEMAP_INDEX = os.getenv("WTTJ_SITEMAP_INDEX", f"{BASE_URL}/sitemaps/index.xml.gz")
LANG          = os.getenv("WTTJ_LANG", "fr")
JOB_PATTERN   = f"/{LANG}/companies/"

HEADERS = {
    "User-Agent": os.getenv(
        "WTTJ_UA",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "gzip, deflate",
    "Accept": "*/*",
}

#  Logging
def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("wttj.ingest.bronze.collect_urls")


#  Téléchargement & parsing XML

def fetch_xml(url: str):
    """Télécharge un sitemap (gzip ou plain XML) et retourne l'arbre XML."""
    try:
        logger.info(f"  Téléchargement : {url}")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        content = resp.content

        # Décompression gzip si nécessaire
        if url.endswith(".gz"):
            try:
                content = gzip.decompress(content)
            except Exception:
                pass

        return ET.fromstring(content)

    except requests.RequestException as e:
        logger.error(f"Erreur téléchargement {url} : {e}")
        return None
    except ET.ParseError as e:
        logger.error(f"Erreur parsing XML {url} : {e}")
        return None


def extract_locs(root):
    """Extrait toutes les URLs <loc> d'un nÅ“ud XML sitemap."""
    return [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text]


#  Collecte des URLs d'offres
def collect_sitemap_urls(query="", entreprise="", ville="", max_results=0, delay=0.5):
    """Parcourt le sitemap WTTJ et collecte les URLs d'offres filtrées."""

    time_start = time.time()
   
    logger.info("Chargement du sitemap principal WTTJ...")
    root = fetch_xml(SITEMAP_INDEX)
    if root is None:
        logger.info("Tentative sans compression...")
        root = fetch_xml(SITEMAP_INDEX.replace(".gz", ""))
    if root is None:
        logger.error("Impossible de charger le sitemap principal.")
        return []

    all_sub = extract_locs(root)
    logger.info(f"{len(all_sub)} sous-sitemaps trouvés au total.")

    # Garder uniquement les sitemaps contenant des offres FR
    job_sitemaps = [
        u for u in all_sub
        if f"/{LANG}/" in u and ("job" in u.lower() or "compan" in u.lower())
    ]
    if not job_sitemaps:
        logger.info("Aucun sous-sitemap /fr/ specifique, scan de tous les sous-sitemaps...")
        job_sitemaps = all_sub

    logger.info(f"{len(job_sitemaps)} sous-sitemaps d'offres a parcourir.")

    all_urls = []

    for i, sitemap_url in enumerate(job_sitemaps, start=1):
        if max_results != 0 and len(all_urls) >= max_results:
            break

        logger.info(f"[{i}/{len(job_sitemaps)}] {sitemap_url}")
        sub_root = fetch_xml(sitemap_url)
        if sub_root is None:
            continue

        locs = extract_locs(sub_root)

        # Garder uniquement les URLs d'offres
        job_urls = [u for u in locs if JOB_PATTERN in u and "/jobs/" in u]

        # Filtres textuels sur l'URL
        if query:
            job_urls = [u for u in job_urls if query.lower() in u.lower()]
        if entreprise:
            job_urls = [u for u in job_urls if f"/companies/{entreprise.lower()}" in u.lower()]
        if ville:
            v = ville.lower()
            job_urls = [u for u in job_urls if f"_{v}" in u.lower() or f"-{v}" in u.lower()]

        new_urls = [u for u in job_urls if u not in all_urls]
        all_urls.extend(new_urls)

        logger.info(f"  -> {len(new_urls)} offres ({len(all_urls)} total)")

        if i < len(job_sitemaps) and delay > 0:
            time.sleep(delay)

    all_urls = all_urls[:max_results] if max_results > 0 else all_urls  

    try:
        storage = get_storage_from_env("bronze", "welcometothejungle")
        storage_key = "sitemap/urls.txt"
        urls_content = "\n".join(all_urls)
        storage.write_bytes(storage_key, urls_content.encode("utf-8"), content_type="text/plain")
        logger.info(f"OK  {len(all_urls)} URLs sauvegardees in storage at {storage_key}")
    except Exception as e:
        logger.error(f"Erreur sauvegarde storage: {e}")
        with open("./urls.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(all_urls))
        logger.info(f"OK  {len(all_urls)} URLs sauvegardees localement dans './urls.txt' (storage failed)")

    time_end = time.time()
    elapsed = time_end - time_start

    return {
        "success": True,
        "message": f"Ingestion des urls de site map terminée)",
        "elapsed_s": elapsed,
        "total_processed": len(all_urls),
    }


# ─────────────────────────────────────────────
#  Point d'entree CLI
# ─────────────────────────────────────────────
def main():
    setup_logging()
    
    logger.info(
        f"Collecte WTTJ via sitemap XML"
    )

    ret = collect_sitemap_urls(
        query="",
        entreprise="",
        ville="",
        max_results=0,
        delay=0.5,
    )

    if ret.get("success", False) is False   :
        logger.warning("Aucune URL collectee. Verifiez vos filtres.")
        sys.exit(0)
    else:
        logger.info(f"OK  {ret['total_processed']} URLs sauvegardees localement dans './urls.txt' (storage failed)")


if __name__ == "__main__":
    main()
