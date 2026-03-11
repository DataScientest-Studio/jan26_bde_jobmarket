import logging

class JSONOnlyFilter(logging.Filter):
    """Filtre qui ne laisse passer que les messages JSON"""
    def filter(self, record):
        # Ne garder que les messages qui ressemblent à du JSON
        return record.getMessage().strip().startswith('{')
