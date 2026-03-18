#!/bin/bash
# Creates the Airflow metadata database in the existing PostgreSQL instance.
# Runs once at container first start (postgres/init scripts are skipped if data volume exists).
set -e
psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE airflow;" 2>/dev/null || echo "Database 'airflow' already exists, skipping."
