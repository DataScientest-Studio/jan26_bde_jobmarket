"""
Modèles de données pour l'API.

Ce package contient tous les modèles Pydantic utilisés pour les requêtes
et réponses des différents endpoints de l'API.
"""


from .base import BaseJobResponse
from .predict import PredictRequest, PredictResponse
from .ingest import IngestResponse, IngestOffersResponse, IngestWTTJResponse, CollectSitemapsResponse, IngestWTTJOptResponse
from .data import MergeDatasetResponse, StatusTrackingResponse, StatusEvolutionResponse, DimLoadResponse, StarSchemaLoadResponse
from .normalize import NormalizeWTTJResponse, NormalizeFTResponse
from .logging import JSONOnlyFilter

__all__ = [
    # Base
    'BaseJobResponse',

    # Predict models
    'PredictRequest',
    'PredictResponse',

    # Ingest models
    'IngestResponse',
    'IngestOffersResponse',
    'IngestWTTJResponse',
    'CollectSitemapsResponse',
    'IngestWTTJOptResponse',

    # Data processing models
    'MergeDatasetResponse',
    'StatusTrackingResponse',
    'StatusEvolutionResponse',
    'DimLoadResponse',
    'StarSchemaLoadResponse',
    'NormalizeWTTJResponse',
    'NormalizeFTResponse',
    'JSONOnlyFilter',
]
