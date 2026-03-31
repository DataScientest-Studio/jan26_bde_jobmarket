
from dataclasses import dataclass
from datetime import datetime

# ==========================
# Bronze datamodels : classes de données immuables pour les données brutes, avant   
# tout nettoyage ou transformation. 
# 
# Elles sont utilisées dans les fonctions d'ingestion pour structurer les données 
# avant de les nettoyer et de les transformer en modèles plus raffinés pour la couche silver.
# Elles ne contiennent aucune logique métier, juste des structures de données.
#
#  Par exemple, pour les données Welcome to the Jungle, on peut avoir des classes comme :
#  - wtt_bronze_datamodels : représente une offre d'emploi brute telle qu'elle est extraite du sitemap ou de l'API, avec tous les champs tels quels.

#  Ces classes sont immuables (frozen=True) pour éviter les mutations accidentelles et améliorer la clarté du code.

# ==========================        

# Immutable dataclasses to avoid accidental mutations and improve code clarity. 
@dataclass(frozen=True)
class RomeItem:
    code: str
    libelle: str


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class WindowStat:
    start: str
    end: str
    total: int

# ==========================
# dataclass for Welcome to the Jungle bronze layer (raw data)
# ==========================
@dataclass(frozen=True)
class wtt_bronze_datamodels:
    source: str
    segment: str
    url: str
    fetched_at: datetime
    status_code: int
    ok: bool
    error: str
    key: str
    initial_data: dict[str, any ]
    job_data: dict[str, any ]
    parser_version: int