# src/config/env.py
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT: Path | None = None
_ENV_LOADED: bool = False


def get_project_root() -> Path:
    """
    Trouve la racine du projet.
    Priorité:
      1) GIT (rev-parse)
      2) Remontée depuis ce fichier jusqu'à trouver pyproject.toml / .git / requirements.txt
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    # 1) Git root
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        _PROJECT_ROOT = Path(root)
        return _PROJECT_ROOT
    except Exception:
        pass

    # 2) Heuristique fichiers marqueurs
    markers = {"pyproject.toml", ".git", "requirements.txt", "setup.cfg"}
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if any((parent / m).exists() for m in markers):
            _PROJECT_ROOT = parent
            return _PROJECT_ROOT

    # Fallback: 2 niveaux au-dessus de src/config/env.py -> (adapter si besoin)
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    return _PROJECT_ROOT


def load_project_env(env_filename: str = ".env", override: bool = False) -> Path:
    """
    Charge le .env depuis la racine du projet une seule fois (idempotent).
    Retourne le chemin du .env (même s'il n'existe pas).
    """
    global _ENV_LOADED
    root = get_project_root()
    env_path = root / env_filename

    if not _ENV_LOADED:
        load_dotenv(env_path, override=override)
        _ENV_LOADED = True

    return env_path


def require_env(key: str) -> str:
    """
    Récupère une variable d'env ou lève une erreur explicite.
    """
    val = os.getenv(key)
    if val is None or val == "":
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val