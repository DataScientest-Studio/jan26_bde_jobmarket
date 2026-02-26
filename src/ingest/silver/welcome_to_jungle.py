""" 
==============
PRE SILVER Layer
==============

Extact relevant data from WTTJ job offer and company information from Json embedded with Appolo format (window.__INITIAL_DATA__) 
stored in raw format in the storage backend.


"""
from __future__ import annotations

from src.config.env import require_env, get_project_root, load_project_env
load_project_env()  # safe à rappeler (idempotent)

import gzip
import hashlib
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from tqdm import tqdm  
import pandas as pd
import html
from bs4 import BeautifulSoup

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.storage.storage import get_storage_from_env
from src.ingest.tools.rate_limiter import RateLimiter
import src.ingest.tools.time_helpers as time_helpers

# ----------------------------
# Logging
# ----------------------------
def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


logger = logging.getLogger("wttj.ingest.silver")

# ================
# Fixe double encoding issue on json by recursive parsing Apollo''s json
# ================

def _fix_double_encoding(text: str) -> str:
    """
    Fix double UTF-8 encoding issues.
    Example: 'M\u00c3\u00a9canique' -> 'Mécanique'
    """
    try:
        # 1. Decode HTML entities (&#39; -> ', &lt; -> <, etc.)
        text = html.unescape(text)
         
        # 2. Encode as latin-1 to get original UTF-8 bytes, then decode as UTF-8
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If it fails, return original text
        return text


def _fix_double_encoded_dict(obj: Any) -> Any:
    """Recursively fix double UTF-8 encoding in dict/list structures."""
    if isinstance(obj, dict):
        return {k: _fix_double_encoded_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_double_encoded_dict(item) for item in obj]
    elif isinstance(obj, str):
        return _fix_double_encoding(obj)
    else:
        return obj

def get_json_field_from_record(record, field_name):
    if isinstance(record[field_name], str):
        data = json.loads(record[field_name])
    else:
        data = record[field_name]  # déjà un dict !
    return data

def clean_html(data):
    if isinstance(data, str):
        # Remove html tag
        soup = BeautifulSoup(data, 'html.parser')
        text = soup.get_text()
        # Remove Html Entities
        return html.unescape(text).strip()
    return data

def find_field_in_json(data, target_field, path=[]):
    """
    Cherche champ récursivement.
    
    >>> find_field_in_json(record, 'name')
    [{'path': ['job_data', 'name'], 'value': 'Stagiaire...'}]
    """
    matches = []
    
    if isinstance(data, dict):
        if target_field in data:
            matches.append({
                'path': path + [target_field],
                'value': data[target_field]
            })
        
        for key, value in data.items():
            matches.extend(find_field_in_json(value, target_field, path + [key]))
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            matches.extend(find_field_in_json(item, target_field, path + [f"[{i}]"]))
    
    return matches


def get_field_or_default(data, field_name, default=None):
    """Premier match, préserve type."""
    matches = find_field_in_json(data, field_name)
    if matches:
        value = matches[0]['value']
        # Préserve listes/tableaux
        if isinstance(value, (list, dict)):
            return value
        return str(value)[:1000]  # Tronque strings longs
    return default

def set_wttj_all_from_json(storage, jsonld_keys):
    dfs = []
    total_jobs = 0
    
    # BARRE PRINCIPALE UNIQUEMENT (position=0)
    with tqdm(
        total=len(jsonld_keys),
        desc="ETL WTTJ",
        position=0,
        leave=True,
        dynamic_ncols=True,  # ← Auto-width
        mininterval=0.1,
        smoothing=0.1
    ) as pbar:
        
        for i, key in enumerate(jsonld_keys):
            # FONCTION SANS tqdm interne !
            df, stats = set_wttj_all_from_json_silent(storage, key)
            dfs.append(df)
            total_jobs += stats["added"]
            
            # Postfix COURT
            postfix = (
                f"J:{total_jobs:,} | "
                f"+{stats['added']:,} | "
                f"{i+1}/{len(jsonld_keys)}"
            )
            
            pbar.set_postfix_str(postfix, refresh=True)
            pbar.update(1)
    
    return pd.concat(dfs, ignore_index=True)


def get_rome_code_from_ml_prediction(name: str, description: str) -> Optional[str]:
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
        response = requests.post(
            url, 
            json=payload,
            timeout=int(os.getenv("ML_TIMEOUT", "30"))
        )
        response.raise_for_status()  # 4xx/5xx → Exception
        
        result = response.json()
        rome_code = result['rome_pred'] if result else None
        rome_labelle = result['rome_label'] if result else None
        return rome_code, rome_labelle
        
    except requests.exceptions.Timeout:
        logger.error("ML API timeout")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("ML API unreachable")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"ML HTTP {e.response.status_code}: {e.response.text[:200]}")
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"ML JSON error: {e}")
        return None
    except Exception as e:
        logger.error(f"ML unexpected: {e}")
        return None



def set_wttj_all_from_json_silent(storage, key):
    """AUCUN print/tqdm → SILENT."""
    data = []
    errors = 0
    
    try:
        obj = storage.get_object_jsonl(key=key)
        # Iterlines to preserve memory load
        for line_bytes in obj["Body"].iter_lines():
            #print(line_bytes[:3000])
            line = line_bytes.decode('utf-8', errors='ignore').strip()
           
            if not line: continue
            
            try:

                # 1. Parser le JSON
                record = json.loads(line)
                #print(f"🔍 Record keys: {list(record.keys())[:5]}")  # DEBUG
                
                # 2. Corriger le double encodage
                record = _fix_double_encoded_dict(record)
                
                job_data = get_json_field_from_record(record, "job_data")
                #print(f"🔍 job_data type: {type(job_data)}, keys: {list(job_data.keys())[:5] if isinstance(job_data, dict) else 'N/A'}")  # DEBUG
                
                initial_data = get_json_field_from_record(record, "initial_data")
                
                # Vérifier que job_data existe
                if not job_data or not isinstance(job_data, dict):
                    print(f"⚠️ job_data invalide ")
                    errors += 1
                    continue
                
                #pprint(initial_data)
                urls_list = job_data.get("urls", [])
                canonical_url = next(
                    (link.get('href', '') for link in urls_list if link.get('kind') == 'canonical'),
                    ''
                )

                name = clean_html(job_data.get("name", ""))
                description= clean_html(job_data.get("description", ""))
                rome_code, rome_label = get_rome_code_from_ml_prediction(name, description)
                profession = get_field_or_default(record, 'profession')

                data.append({
                    "wttj_reference": job_data.get("wttj_reference"),
                    "reference": job_data.get("reference"),                    
                    "name": name,
                    "description": description,
                    "profile": clean_html(job_data.get("profile")),

                    "salary_min": job_data.get("salary_min"),
                    "salary_max": job_data.get("salary_max"),
                    "salary_currency": job_data.get("salary_currency"),                    
                    "education_level": job_data.get("education_level"),
                    "company_summary": job_data.get("company_summary"),
                    "company_description": job_data.get("company_description"),

                    "updated_at": job_data.get("updated_at"),
                    "published_at": job_data.get("published_at"),
                    "archived_at": job_data.get("archived_at"),
                    
                    "contract_duration_min": job_data.get("contract_duration_min"),                    
                    "remote": job_data.get("remote"),
                    "ats": job_data.get("ats"),
                    "contract_duration_max": job_data.get("contract_duration_max"),
                    "experience_level": job_data.get("experience_level"),
                    "contract_type": job_data.get("contract_type"),
                    
                    "urls": urls_list,  
                    "canonical_url" : canonical_url,
                    "skills": job_data.get("skills", [""]),                    
                    "key_missions": job_data.get("key_missions", [""]),
                    "offices": job_data.get("offices", [""]),

                    "sectors" : get_field_or_default(record, 'sectors', []),
                    "profession" : profession,
                    "rome_code": rome_code ,
                    "rome_label": rome_label

                })
            except Exception as e:
                errors += 1
                print(f"⚠️ Error parsing line in {key}: {e}")
                # DEBUG: afficher la ligne problématique
                if errors <= 3:  # Limite à 3 exemples
                    print(f"   Line: {line[:200]}...")
                        
        
        df = pd.DataFrame(data)
        return df, {"added": len(df), "errors": errors}
    
    except Exception as e:
        print(f"⚠️ Error reading key {key}: {e} | {e.__class__.__name__}")
        return pd.DataFrame(), {"added": 0, "errors": 1}


# ----------------------------
# Main (new / resume / incremental)
# ----------------------------
def main() -> None:
    setup_logging()

    dt = os.getenv("DT") or datetime.now().date().isoformat()

    provided_run_id = (os.getenv("WTTJ_RUN_ID") or "").strip()
    resume_from_run_id = (os.getenv("WTTJ_RESUME_FROM_RUN_ID") or "").strip()

    run_id = time_helpers.run_id_utc()
   
    # Storage (local / S3) — mêmes variables que votre storage.py,
    # mais ici on passe un root dédié WTTJ + un prefix dédié.
    storage = get_storage_from_env(
        os.getenv("WTTJ_DATA_DIR", "data/welcometothejungle"),
        os.getenv("S3_PREFIX_WTTJ", "welcometothejungle"),
    )

    logger.info("Run start | dt=%s | run_id=%s", dt, run_id)

    silver_jobs_prefix = f"silver/dt={dt}/run_id={run_id}/segment=jobs/"
    source_bronze_prefix = "bronze/dt=2026-02-18/run_id=20260217T181105Z/segment=jobs_raw"

    # Get Json ld keys
    jsonld_keys = storage.list_keys(source_bronze_prefix)

    df= set_wttj_all_from_json(storage=storage, jsonld_keys=jsonld_keys)

    # Fix mixed types in object columns (dicts/lists → JSON strings, NaN → None)
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Debug: print(f"{col}: {df[col].map(type).value_counts().head()}")
        df[col] = df[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) 
            else (str(x) if pd.notna(x) else None)
        )

    # Save parquet in silver layer
    storage.write_parquet(silver_jobs_prefix + "wttj_jobs.parquet", df)

    # We then call the ingest_segment function for both jobs and companies.

    logger.info("Run done |  dt=%s | run_id=%s", dt, run_id)


if __name__ == "__main__":
    main()