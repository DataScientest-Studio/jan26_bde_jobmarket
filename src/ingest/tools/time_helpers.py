from datetime import datetime, timezone 

# ----------------------------
# Time helpers
# ----------------------------
def utc_now_iso() -> str:
    """ Return the current UTC time in ISO 8601 format. """
    return datetime.now(timezone.utc).isoformat()


def run_id_utc() -> str:
    """ Generate a run ID based on the current UTC time in a compact format. """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
