"""
Modèle de base commun à toutes les réponses de l'API jobmarket.

Garantit que success, message et records_count sont toujours présents,
quel que soit l'endpoint ou le mode d'exécution (sync / background).
"""
from typing import Optional
from pydantic import BaseModel, Field


class BaseJobResponse(BaseModel):
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message descriptif du résultat")
    records_count: Optional[int] = Field(
        None,
        description=(
            "Nombre d'enregistrements traités. "
            "Toujours None en mode background (tâche non encore exécutée). "
            "Champ source selon l'endpoint : written (FT), total_written (WTTJ), "
            "len(files) (normalize), total_offers (merge), "
            "total_status_records (status_tracking), rows_timeline (status_evolution), "
            "upserted (dim), fact_rows (star schema)."
        ),
    )
