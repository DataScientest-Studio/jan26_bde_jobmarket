# Configuration pgAdmin

## Fichier servers.json

Ce fichier contient les configurations de connexion aux serveurs PostgreSQL dans pgAdmin.

**⚠️ Ce fichier contient des mots de passe en clair et ne doit PAS être commité dans Git.**

### Première configuration

1. Copiez le fichier exemple:
   ```bash
   cp servers.json.example servers.json
   ```

2. Éditez `servers.json` et remplacez `CHANGEME` par le vrai mot de passe

3. Le fichier sera automatiquement chargé au démarrage de pgAdmin

### Variables à configurer

- `Password`: Mot de passe PostgreSQL (doit correspondre à `POSTGRES_PASSWORD` dans `.env`)
- `Username`: Utilisateur PostgreSQL (par défaut: `jobuser`)
- `Host`: Nom du service Docker (par défaut: `postgres`)

### Sécurité

✅ `servers.json` est dans `.gitignore`
✅ `servers.json.example` peut être commité (pas de secrets)
❌ Ne commitez jamais `servers.json` avec le vrai mot de passe
