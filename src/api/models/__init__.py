"""
Modèles de données pour l'API.

Ce package contient tous les modèles Pydantic utilisés pour les requêtes
et réponses des différents endpoints de l'API.
"""
from .predict import PredictRequest, PredictResponse
from .ingest import IngestResponse, IngestOffersResponse, IngestWTTJResponse
from .data import MergeDatasetResponse

__all__ = [
    # Predict models
    'PredictRequest',
    'PredictResponse',
    
    # Ingest models
    'IngestResponse',
    'IngestOffersResponse',
    'IngestWTTJResponse',
    
    # Data processing models
    'MergeDatasetResponse',
]
