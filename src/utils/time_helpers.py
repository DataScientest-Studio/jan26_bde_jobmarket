from datetime import datetime, timedelta, timezone

def utc_dt_str() -> str:
    """ Return current date in UTC as a string in YYYY-MM-DD format, suitable for partitioning. """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def utc_run_id() -> str:
    """ Generate a unique run ID based on the current UTC timestamp, in ISO format without separators. """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def to_iso_z(dt: datetime) -> str:
    """ Convert a datetime to ISO 8601 format with 'Z' suffix for UTC timezone. """
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

def format_eta(seconds: float) -> str:
    """ Format a duration in seconds into a human-readable string HH:MM:SS. If the input is zero or negative, return "00:00:00".
        This is used for logging elapsed time and estimated time remaining in the ingestion process.
      """
    if seconds <= 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
