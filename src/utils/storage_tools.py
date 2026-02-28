def get_last_dt_from_storage(storage, prefix: str) -> str:
    """Get last dt=... from storage keys under given prefix, return as string YYYY-MM-DD"""
    prefixes = storage.list_prefixes(prefix)
    dts = [p.split("dt=")[-1].split("/")[0] for p in prefixes if "dt=" in p]
    return max(dts) if dts else None
