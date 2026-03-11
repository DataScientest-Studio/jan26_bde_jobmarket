"""
Utilitaires de traitement de texte.

Ce module fournit des fonctions pour nettoyer, normaliser et extraire
des informations à partir de texte brut, HTML et données structurées.
"""
import html
import re
from typing import List, Optional

from bs4 import BeautifulSoup


def clean_html(text: Optional[str]) -> str:
    """
    Nettoie le HTML et les entités (optimisé).
    
    Args:
        text: Texte potentiellement contenant du HTML
        
    Returns:
        Texte nettoyé sans balises HTML ni entités
        
    Examples:
        >>> clean_html("<p>Bonjour &eacute;tudiant</p>")
        'Bonjour étudiant'
        
        >>> clean_html("Simple texte")
        'Simple texte'
    """
    if text is None or not isinstance(text, str) or not text:
        return ""
    
    # Quick check: only use BeautifulSoup if HTML tags detected
    # This avoids slow BeautifulSoup calls on plain text
    if '<' in text and '>' in text:
        # Remove HTML tags with lxml (2-3x faster than html.parser)
        soup = BeautifulSoup(text, 'lxml')
        text = soup.get_text()
    
    # Remove HTML entities (always, they can exist without tags)
    # "&#233;" → "é"
    text = html.unescape(text)
    
    # Normalise whitespace
    text = ' '.join(text.split())
    
    return text.strip()

#def clean_html(data):
#    if isinstance(data, str):
#        # Remove html tag
#        soup = BeautifulSoup(data, 'html.parser')
#        text = soup.get_text()
#        # Remove Html Entities
#        return html.unescape(text).strip()
#     return data



def normalize_text(text: Optional[str]) -> str:
    """
    Normalisation rapide de texte avec nettoyage HTML et caractères de contrôle.
    
    Args:
        text: Texte à normaliser
        
    Returns:
        Texte normalisé et nettoyé
        
    Examples:
        >>> normalize_text("<p>Test   avec espaces</p>")
        'Test avec espaces'
        
        >>> normalize_text("Ligne1\\x00\\x01Ligne2")
        'Ligne1Ligne2'
    """
    if text is None or (isinstance(text, str) and not text):
        return ""
    
    # Clean HTML first (with fast HTML tag detection)
    text = clean_html(text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove control characters (optimisé avec regex au lieu de ord())
    # Sinon cette boucle est très lente sur de grands textes
    # Garde uniquement les caractères imprimables + \n\t
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def normalize_list_to_strings(list_data) -> List[str]:
    """
    Normalises any list field (skills, offices, missions, sectors, etc.) to a consistent List[str] format.
    
    Gère les formats suivants:
    - Chaîne de caractères avec séparateur virgule
    - Liste de chaînes de caractères
    - Liste de dictionnaires avec clé "name"
    - Autres types convertis en string
    
    Args:
        skills_data: Données de compétences (str, list, dict, etc.)
        
    Returns:
        Liste de compétences (strings)
        
    Examples:
        >>> normalize_list_to_strings("Python, JavaScript, SQL")
        ['Python', 'JavaScript', 'SQL']
        
        >>> normalize_list_to_strings(["Python", "JavaScript"])
        ['Python', 'JavaScript']
        
        >>> normalize_list_to_strings([{"name": "Python"}, {"name": "SQL"}])
        ['Python', 'SQL']
        
        >>> normalize_list_to_strings(None)
        []
    """
    # Handle None, empty strings, or empty lists
    if list_data is None or (isinstance(list_data, str) and not list_data.strip()):
        return []
    
    # If it's a comma-separated string
    if isinstance(list_data, str):
        return [s.strip() for s in list_data.split(",") if s.strip()]
    
    # If it's already a list of strings or dicts
    if isinstance(list_data, list):
        items = []
        for item in list_data:
            if isinstance(item, str):
                items.append(item.strip())
            elif isinstance(item, dict):
                name = item.get("name", "")
                if isinstance(name, str) and name.strip():
                    items.append(name.strip())
        return items
    
    # Other format? Convert to string
    return [str(list_data).strip()]
