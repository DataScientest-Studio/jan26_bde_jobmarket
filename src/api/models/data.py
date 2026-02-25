"""
Modèles de réponse pour les opérations de traitement de données.
"""
from typing import Optional
from pydantic import BaseModel, Field


class MergeDatasetResponse(BaseModel):
    """Résultat d'une opération de fusion de datasets"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    output_key: Optional[str] = Field(None, description="Clé de stockage du dataset fusionné", example="merged_dataset_20260225_143000.parquet")
    output_format: Optional[str] = Field(None, description="Format du fichier de sortie", example="parquet")
    ft_prefix: Optional[str] = Field(None, description="Préfixe des données France Travail utilisées")
    wttj_prefix: Optional[str] = Field(None, description="Préfixe des données WTTJ utilisées")
    total_offers: Optional[int] = Field(None, description="Nombre total d'offres fusionnées", example=150000)
    ft_offers: Optional[int] = Field(None, description="Nombre d'offres France Travail", example=120000)
    wttj_offers: Optional[int] = Field(None, description="Nombre d'offres WTTJ", example=30000)
    offers_with_rome: Optional[int] = Field(None, description="Nombre d'offres avec code ROME", example=140000)
    unique_rome_codes: Optional[int] = Field(None, description="Nombre de codes ROME uniques", example=532)
    elapsed_s: Optional[float] = Field(None, description="Durée totale en secondes", example=1200.5)
    error: Optional[str] = Field(None, description="Message d'erreur si échec")
