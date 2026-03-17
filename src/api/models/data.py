"""
Modèles de réponse pour les opérations de traitement de données.
"""
from typing import Any, Dict, Optional
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


class StatusTrackingResponse(BaseModel):
    """Résultat d'un run de suivi du cycle de vie des offres d'emploi"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    mode: Optional[str] = Field(None, description="Mode d'exécution : 'incremental' ou 'global_recalc'", examples=["incremental"])
    output_key: Optional[str] = Field(None, description="Clé de stockage du fichier de statuts (mode incremental)", examples=["dt=2026-03-17/segment=offer_status/offer_status.parquet"])
    run_json_key: Optional[str] = Field(None, description="Clé du fichier de métadonnées run.json (mode incremental)")
    merged_dt: Optional[str] = Field(None, description="Date du dataset mergé de référence (mode incremental)", examples=["2026-03-17"])
    total_status_records: Optional[int] = Field(None, description="Nombre total de lignes de statut produites", examples=[125000])
    dates_processed: Optional[int] = Field(None, description="Nombre de dates traitées (mode global_recalc)", examples=[30])
    status_stats: Optional[Dict[str, Any]] = Field(None, description="Statistiques par source (published / unpublished / reappeared)")
    elapsed_sec: Optional[float] = Field(None, description="Durée d'exécution en secondes", examples=[12.4])
    error: Optional[str] = Field(None, description="Message d'erreur si échec")


class StatusEvolutionResponse(BaseModel):
    """Résultat d'un run de génération des datasets analytiques d'évolution de statut"""
    success: bool = Field(..., description="Succès de l'opération")
    message: Optional[str] = Field(None, description="Message descriptif du résultat")
    job_id: Optional[str] = Field(None, description="Identifiant unique du run")
    analysis_dt: Optional[str] = Field(None, description="Date de référence de l'analyse", examples=["2026-03-17"])
    mode: Optional[str] = Field(None, description="Mode d'exécution : 'incremental' ou 'recompute_all'", examples=["incremental"])
    complete_key: Optional[str] = Field(None, description="Clé du dataset complet (offres + statuts)")
    timeline_key: Optional[str] = Field(None, description="Clé du dataset timeline de statuts")
    run_key: Optional[str] = Field(None, description="Clé du fichier de métadonnées run.json")
    rows_complete: Optional[int] = Field(None, description="Nombre de lignes dans le dataset complet", examples=[125000])
    rows_timeline: Optional[int] = Field(None, description="Nombre de lignes dans le dataset timeline", examples=[375000])
    elapsed_sec: Optional[float] = Field(None, description="Durée d'exécution en secondes", examples=[18.5])
    error: Optional[str] = Field(None, description="Message d'erreur si échec")
