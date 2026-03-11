-- Table pour les logs d'ingestion (remplace Loki/Promtail)
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    endpoint VARCHAR(50) NOT NULL,
    level VARCHAR(10) NOT NULL DEFAULT 'INFO',
    task_id VARCHAR(100),
    message TEXT NOT NULL,
    
    -- Métriques communes (colonnes dédiées pour faciliter les requêtes Grafana)
    duration_sec NUMERIC(10, 3),
    records_count BIGINT,
    error_count INTEGER DEFAULT 0,
    
    -- Métadonnées spécifiques par endpoint (flexibilité)
    extra_metadata JSONB
);

-- Index pour performance des requêtes Grafana
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_timestamp ON ingestion_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_endpoint ON ingestion_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_level ON ingestion_logs(level);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_task_id ON ingestion_logs(task_id);

-- Fonction de nettoyage automatique (rétention 30 jours)
CREATE OR REPLACE FUNCTION cleanup_old_ingestion_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ingestion_logs
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Optionnel : Créer un trigger ou schedule pour cleanup automatique
-- Pour l'instant, le cleanup peut être appelé manuellement ou via cron
COMMENT ON FUNCTION cleanup_old_ingestion_logs() IS 'Supprime les logs d''ingestion de plus de 30 jours';
COMMENT ON TABLE ingestion_logs IS 'Logs d''ingestion centralisés (endpoints: rome_metiers, france_travail_offers, wttj, merge_datasets)';
