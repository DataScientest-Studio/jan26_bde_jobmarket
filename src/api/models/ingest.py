"""
Modèles de réponse pour les opérations d'ingestion de données.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Résultat d'une opération d'ingestion de données"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    key: Optional[str] = Field(None, description="Clé de stockage des données", example="bronze/rome/rome_metiers.jsonl")
    records_count: Optional[int] = Field(None, description="Nombre total de codes ROME", example=532)
    records_written: Optional[int] = Field(None, description="Nombre d'enregistrements écrits", example=532)
    error: Optional[str] = Field(None, description="Message d'erreur si échec")


class IngestOffersResponse(BaseModel):
    """Résultat d'une opération d'ingestion des offres d'emploi"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    run_id: Optional[str] = Field(None, description="Identifiant unique du run", example="20260223T120000Z")
    run_key: Optional[str] = Field(None, description="Clé des métadonnées du run")
    rome_processed: Optional[int] = Field(None, description="Nombre de codes ROME traités", example=532)
    calls: Optional[int] = Field(None, description="Nombre d'appels API effectués", example=1500)
    written: Optional[int] = Field(None, description="Nombre total d'offres écrites", example=15000)
    elapsed_s: Optional[float] = Field(None, description="Durée de l'ingestion en secondes", example=3600.5)
    errors: Optional[int] = Field(None, description="Nombre d'erreurs rencontrées", example=0)
    error: Optional[str] = Field(None, description="Message d'erreur si échec")


class IngestWTTJResponse(BaseModel):
    """Résultat d'une opération d'ingestion Welcome to the Jungle"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    run_id: Optional[str] = Field(None, description="Identifiant unique du run", example="20260223T120000Z")
    dt: Optional[str] = Field(None, description="Date de l'ingestion", example="2026-02-23")
    mode: Optional[str] = Field(None, description="Mode d'ingestion (new, resume, incremental)", example="new")
    total_processed: Optional[int] = Field(None, description="Nombre total d'URLs traitées", example=1500)
    total_written: Optional[int] = Field(None, description="Nombre total de records écrits", example=1500)
    elapsed_s: Optional[float] = Field(None, description="Durée totale en secondes", example=600.5)
    jobs: Optional[Dict[str, Any]] = Field(None, description="Statistiques segment jobs")
    companies: Optional[Dict[str, Any]] = Field(None, description="Statistiques segment companies")
    error: Optional[str] = Field(None, description="Message d'erreur si échec")


class CollectSitemapsResponse(BaseModel):
    """Résultat de la collecte des URLs depuis les sitemaps WTTJ"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    urls_count: int = Field(..., description="Nombre d'URLs collectées", example=80418)
    storage_key: Optional[str] = Field(None, description="Clé de stockage des URLs", example="sitemap/urls.txt")
    elapsed_s: Optional[float] = Field(None, description="Durée de la collecte en secondes", example=45.2)
    error: Optional[str] = Field(None, description="Message d'erreur si échec")


class IngestWTTJOptResponse(BaseModel):
    """Résultat d'une opération d'ingestion Welcome to the Jungle optimisée"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    run_id: Optional[str] = Field(None, description="Identifiant unique du run", example="20260223T120000Z")
    dt: Optional[str] = Field(None, description="Date de l'ingestion", example="2026-02-23")
    mode: Optional[str] = Field(None, description="Mode d'ingestion (new, resume, incremental)", example="new")
    urls_total: Optional[int] = Field(None, description="Nombre total d'URLs à traiter", example=10000)
    urls_processed: Optional[int] = Field(None, description="Nombre d'URLs traitées", example=10000)
    urls_ok: Optional[int] = Field(None, description="Nombre d'URLs traitées avec succès", example=9950)
    urls_ko: Optional[int] = Field(None, description="Nombre d'URLs en erreur", example=50)
    records_written: Optional[int] = Field(None, description="Nombre de records JSONL écrits", example=9950)
    elapsed_s: Optional[float] = Field(None, description="Durée totale en secondes", example=1800.5)
    storage_prefix: Optional[str] = Field(None, description="Préfixe de stockage", example="dt=2026-02-23/run_id=...")
    error: Optional[str] = Field(None, description="Message d'erreur si échec")
