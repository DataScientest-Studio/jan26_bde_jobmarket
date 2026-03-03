-- ============================================================================
-- 000_init_user_db.sql
-- Initialisation du rôle et de la base (exécuté en premier, avant les autres)
-- 
-- IMPORTANT: 
-- - La base 'jobmarket' est automatiquement créée par Docker via POSTGRES_DB
-- - L'utilisateur 'jobuser' est automatiquement créé par Docker via POSTGRES_USER
-- - Ce script s'exécute DANS la base 'jobmarket' déjà créée
-- - Il configure uniquement les DROITS de l'utilisateur
-- ============================================================================

-- Vérification : on est bien dans la bonne base
DO $$
BEGIN
  IF current_database() != 'jobdb' THEN
    RAISE EXCEPTION 'Ce script doit s''exécuter dans la base jobdb, actuellement dans: %', current_database();
  END IF;
  
  -- Vérifier que l'utilisateur jobuser existe (créé par Docker)
  IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'jobuser') THEN
    RAISE EXCEPTION 'L''utilisateur jobuser n''existe pas. Vérifiez POSTGRES_USER dans .env';
  END IF;
  
  RAISE NOTICE 'Base: % | Utilisateur: jobuser existe ✓', current_database();
END
$$;

-- Accorder les permissions minimales nécessaires
-- 1. Connexion à la base de données
GRANT CONNECT ON DATABASE jobdb TO jobuser;

-- 2. Droits sur le schéma public (pour les scripts 001, 002 et l'applicatif)
GRANT USAGE ON SCHEMA public TO jobuser;           -- Utiliser le schéma
GRANT CREATE ON SCHEMA public TO jobuser;          -- Créer des tables (scripts init)

-- 3. Droits sur les tables existantes et futures
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO jobuser;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO jobuser;

-- 4. Droits par défaut pour les objets futurs
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO jobuser;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO jobuser;

COMMENT ON ROLE jobuser IS 'Application user for jobmarket data pipeline';
