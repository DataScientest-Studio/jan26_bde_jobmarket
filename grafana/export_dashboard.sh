#!/bin/bash
# Script pour exporter tous les dashboards Grafana dans grafana/backup/
# Dépendances : curl, jq

GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="admin"
BACKUP_DIR="./grafana/backup"

mkdir -p "$BACKUP_DIR"

# Récupérer tous les UIDs de dashboards
uids=$(curl -s -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/search?query=" | jq -r '.[] | select(.type=="dash-db") | .uid')

for uid in $uids; do
    name=$(curl -s -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/dashboards/uid/$uid" | jq -r '.dashboard.title' | tr ' ' '_' | tr -dc 'A-Za-z0-9_-')
    curl -s -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/dashboards/uid/$uid" | jq '.dashboard' > "$BACKUP_DIR/${name}_${uid}.json"
done

echo "Export terminé. Dashboards sauvegardés dans $BACKUP_DIR"