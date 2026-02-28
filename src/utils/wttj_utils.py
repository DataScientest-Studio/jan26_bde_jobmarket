from typing import List, Any, Optional
import html
import json

def find_field_in_json(data, target_field, path=[]):
    """
    Cherche champ récursivement.
    >>> find_field_in_json(record, 'name')
    [{'path': ['job_data', 'name'], 'value': 'Stagiaire...'}]
    """
    matches = []
    if isinstance(data, dict):
        if target_field in data:
            matches.append({
                'path': path + [target_field],
                'value': data[target_field]
            })
        for key, value in data.items():
            matches.extend(find_field_in_json(value, target_field, path + [key]))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            matches.extend(find_field_in_json(item, target_field, path + [f"[{i}]"]))
    return matches

def get_field_or_default(data, field_name, default=None):
    """Premier match, préserve type."""
    matches = find_field_in_json(data, field_name)
    if matches:
        value = matches[0]['value']
        # Préserve listes/tableaux
        if isinstance(value, (list, dict)):
            return value
        return str(value)[:1000]  # Tronque strings longs
    return default

# ================
# Fixe double encoding issue on json by recursive parsing Apollo''s json
# ================

def _fix_double_encoding(text: str) -> str:
    """
    Fix double UTF-8 encoding issues.
    Example: 'M\u00c3\u00a9canique' -> 'Mécanique'
    """
    try:
        # 1. Decode HTML entities (&#39; -> ', &lt; -> <, etc.)
        text = html.unescape(text)
         
        # 2. Encode as latin-1 to get original UTF-8 bytes, then decode as UTF-8
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If it fails, return original text
        return text


def _fix_double_encoded_dict(obj: Any) -> Any:
    """Recursively fix double UTF-8 encoding in dict/list structures."""
    if isinstance(obj, dict):
        return {k: _fix_double_encoded_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_double_encoded_dict(item) for item in obj]
    elif isinstance(obj, str):
        return _fix_double_encoding(obj)
    else:
        return obj


def get_json_field_from_record(record, field_name):
    if isinstance(record[field_name], str):
        data = json.loads(record[field_name])
    else:
        data = record[field_name]  # déjà un dict !
    return data

def find_field_in_json(data, target_field, path=[]):
    """
    Cherche champ récursivement.
    
    >>> find_field_in_json(record, 'name')
    [{'path': ['job_data', 'name'], 'value': 'Stagiaire...'}]
    """
    matches = []
    
    if isinstance(data, dict):
        if target_field in data:
            matches.append({
                'path': path + [target_field],
                'value': data[target_field]
            })
        
        for key, value in data.items():
            matches.extend(find_field_in_json(value, target_field, path + [key]))
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            matches.extend(find_field_in_json(item, target_field, path + [f"[{i}]"]))
    
    return matches
