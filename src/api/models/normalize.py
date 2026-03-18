from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseJobResponse


class NormalizeWTTJResponse(BaseJobResponse):
    """Résultat d'une normalisation des offres WTTJ bronze → silver.

    Hérite de BaseJobResponse : success, message, records_count (= len(files)).
    """
    job_id: str
    status: str
    dt: Optional[str] = None
    format: str
    files: List[str]
    errors: int


class NormalizeFTResponse(BaseJobResponse):
    """Résultat d'une normalisation des offres France Travail bronze → silver.

    Hérite de BaseJobResponse : success, message, records_count (= len(files)).
    """
    job_id: str
    status: str
    dt: Optional[str] = None
    format: str
    files: List[str]
    errors: int
