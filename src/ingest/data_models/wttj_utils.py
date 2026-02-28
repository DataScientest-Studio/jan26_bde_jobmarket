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
