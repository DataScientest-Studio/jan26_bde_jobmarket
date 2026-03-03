"""
Utilitaire de résolution automatique des préfixes de données.

Ce module fournit des fonctions pour détecter automatiquement
les préfixes de données les plus récents dans une structure dt=date.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def find_latest_data_prefix(storage, base_prefix: str, segment: str = "") -> Optional[str]:
    """
    Détecte automatiquement le préfixe de données le plus récent basé sur la structure dt=date.
    Liste directement au niveau base_prefix/segment pour trouver les répertoires dt=.
    
    Args:
        storage: Instance de storage (avec client S3)
        base_prefix: Préfixe de base (ex: "bronze", "silver")
        segment: Segment optionnel (ex: "jobs_raw", "jobs", "offers")
        
    Returns:
        Préfixe complet avec la date la plus récente ou None
        
    Examples:
        >>> # Avec segment
        >>> find_latest_data_prefix(storage, "bronze", "jobs_raw")
        'bronze/jobs_raw/dt=2026-02-18'
        
        >>> # Sans segment
        >>> find_latest_data_prefix(storage, "silver", "")
        'silver/dt=2026-02-22'
    """
    try:
        # Construire le préfixe de recherche: base_prefix/segment/
        if(segment and not base_prefix):
            search_prefix = f"{segment}"
        else:
            search_prefix = f"{base_prefix}/{segment}/" if segment else f"{base_prefix}/"
        full_search_prefix = storage._full_key(search_prefix)
        
        logger.info(f"🔍 Recherche des dates sous: {search_prefix}")
        
        # Lister les répertoires de 1er niveau avec délimiteur
        response = storage.client.list_objects_v2(
            Bucket=storage.bucket,
            Prefix=full_search_prefix,
            Delimiter='/'
        )
        
        # Extraire les dates des CommonPrefixes (répertoires dt=)
        dates = []
        if 'CommonPrefixes' in response:
            for prefix_info in response['CommonPrefixes']:
                prefix = prefix_info['Prefix']
                # Extraire le nom du répertoire (dt=YYYY-MM-DD)
                dir_name = prefix.replace(full_search_prefix, '').rstrip('/')
                
                if dir_name.startswith('dt='):
                    date_str = dir_name.replace('dt=', '')
                    dates.append(date_str)
                    logger.debug(f"   Trouvé: {dir_name}")
        
        if not dates:
            logger.warning(f"   ⚠️ Aucune date dt= trouvée sous {search_prefix}")
            return None
        
        # Trier les dates en ordre décroissant (plus récent d'abord)
        dates.sort(reverse=True)
        latest_date = dates[0]
        
        logger.info(f"   ✓ Date la plus récente: {latest_date}")
        logger.info(f"   (parmi {len(dates)} dates trouvées)")
        
        # Construire le préfixe complet
        complete_prefix = f"{base_prefix}/{segment}/dt={latest_date}" if segment else f"{base_prefix}/dt={latest_date}"
        
        # Vérifier qu'il y a bien des fichiers sous ce préfixe
        test_response = storage.client.list_objects_v2(
            Bucket=storage.bucket,
            Prefix=storage._full_key(complete_prefix),
            MaxKeys=1
        )
        
        if test_response.get('KeyCount', 0) > 0:
            logger.info(f"   ✓ Préfixe validé: {complete_prefix}")
            return complete_prefix
        else:
            logger.warning(f"   ⚠️ Aucun fichier trouvé sous {complete_prefix}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la détection du préfixe: {e}")
        return None
