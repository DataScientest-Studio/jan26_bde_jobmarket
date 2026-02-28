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
