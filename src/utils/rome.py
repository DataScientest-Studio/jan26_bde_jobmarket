# ================
# ROME Prediction
# ================

import os

from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)
import requests
import logging
from typing import List, Optional
import json
import os

logger = logging.getLogger(__name__)

def get_rome_code_from_ml_prediction_via_api(name: str, description: str) -> Optional[str]:
    """
    Récupère code ROME principal depuis ML API.
    
    Args:
        name: Titre poste
        description: Description complète
        
    Returns:
        Code ROME principal ou None
    """
    payload = {
        "intitule": name or "",
        "description": description or ""
    }
    
    try:
        # API depuis .env
        api_url = os.getenv("ML_HOST_API", "http://localhost:8000")
        endpoint = os.getenv("ML_ENDPOINT", "predict")
        url = f"{api_url}/{endpoint}"
        
        #logger.info(f"ML predict: {name[:50]}...")
        # Timeout split : connect court, read long pour absorber le cold start (chargement du modèle)
        response = requests.post(
            url,
            json=payload,
            timeout=(
                int(os.getenv("ML_CONNECT_TIMEOUT", "10")),   # connexion TCP
                int(os.getenv("ML_READ_TIMEOUT", "120")),     # attente réponse (cold start modèle)
            )
        )
        response.raise_for_status()  # 4xx/5xx → Exception
        
        result = response.json()
        rome_code = result['rome_pred'] if result else None
        rome_labelle = result['rome_label'] if result else None
        return rome_code, rome_labelle
        
    except requests.exceptions.Timeout:
        logger.error("ML API timeout")
        return None, None
    except requests.exceptions.ConnectionError:
        logger.error("ML API unreachable")
        return None, None
    except requests.exceptions.HTTPError as e:
        logger.error(f"ML HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None, None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"ML JSON error: {e}")
        return None, None
    except Exception as e:
        logger.error(f"ML unexpected: {e}")
        return None, None
