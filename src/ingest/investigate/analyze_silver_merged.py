from src.config.env import load_project_env
load_project_env()  # safe à rappeler (idempotent)

from src.storage.storage import get_storage_from_env
import src.utils.merge_dataset_utils as merge_utils
import logging
logger = logging.getLogger(__name__)

storage_wttj = get_storage_from_env("silver", "merged")
df = merge_utils.read_wttj_parquet_file_to_df(storage_wttj,"")
if df is not None and not df.empty:
    merge_utils.print_statistics(df)
else    :
    logger.warning("⚠️ Aucune donnée chargée pour les statistiques FT/WTTJ fusionnées")