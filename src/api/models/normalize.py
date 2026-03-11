from pydantic import BaseModel
from typing import List

class NormalizeWTTJResponse(BaseModel):
    job_id: str
    status: str
    dt: str | None = None
    format: str
    files: List[str]
    errors: int


class NormalizeFTResponse(BaseModel):
    job_id: str
    status: str
    dt: str | None = None
    format: str
    files: List[str]
    errors: int
