"""Export all Grafana dashboards to grafana/backup/"""
import json
import os
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")
BACKUP_DIR = Path(__file__).parent / "backup"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.auth = (GRAFANA_USER, GRAFANA_PASSWORD)

dashboards = session.get(f"{GRAFANA_URL}/api/search?query=&type=dash-db").json()
print(f"Found {len(dashboards)} dashboard(s)")

for d in dashboards:
    uid = d["uid"]
    detail = session.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}").json()
    title = detail["dashboard"]["title"]
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", title)
    path = BACKUP_DIR / f"{safe_name}_{uid}.json"
    path.write_text(json.dumps(detail["dashboard"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ok {title} ({uid}) -> {path.name}")

print(f"\nExport termine -- {len(dashboards)} dashboard(s) dans {BACKUP_DIR}")
