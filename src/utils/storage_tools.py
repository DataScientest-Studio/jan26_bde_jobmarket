from typing import Optional
from src.storage.storage import Storage
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_last_dt_from_storage(storage: Storage, prefix: str) -> Optional[str]:
    """Get last dt=... from storage keys under given prefix, return as string YYYY-MM-DD"""
    prefixes = storage.list_prefixes(prefix)
    dts = [p.split("dt=")[-1].split("/")[0] for p in prefixes if "dt=" in p]
    return max(dts) if dts else None

def load_parquet_dataset(storage: Storage, key: str) -> Optional[pd.DataFrame]:
    """
    Load a parquet file from storage.
    
    Args:
        storage: Storage instance (S3Storage or LocalStorage)
        key: Logical key to the parquet file
    
    Returns:
        DataFrame or None if file not found
    """
    try:
        logger.info(f"   📖 Loading parquet from: {key}")
        df = storage.read_parquet(key)
        logger.info(f"   ✓ Loaded {len(df):,} rows")
        return df
    except FileNotFoundError:
        logger.info(f"   ℹ️ File not found: {key}")
        return None
    except Exception as e:
        logger.warning(f"   ⚠️ Error loading parquet: {e}")
        return None


def save_parquet_dataset(storage: Storage, df: pd.DataFrame, key: str) -> bool:
    """
    Save a parquet file to storage with updated status fields.
    
    Args:
        storage: Storage instance (S3Storage or LocalStorage)
        df: DataFrame with status and unpublished_at columns
        key: Logical key for output parquet file
    
    Returns:
        True if save succeeded, False otherwise
    """
    try:
        logger.info(f"   💾 Saving parquet to: {key}")
        storage.write_parquet(key, df)
        logger.info(f"   ✓ Saved {len(df):,} rows")
        return True
    except Exception as e:
        logger.error(f"   ❌ Error saving parquet: {e}")
        return False
