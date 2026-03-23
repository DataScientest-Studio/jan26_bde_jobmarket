"""
Connexion PostgreSQL partagée entre toutes les pages.
Utilise st.cache_resource pour une connexion unique par session Streamlit.
"""
import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# Idempotent get_engine() : une seule connexion par session Streamlit, partagée entre les pages
@st.cache_resource
def get_engine():
    dsn = os.getenv("JOBSTORE_DSN", "")
    return create_engine(dsn, pool_pre_ping=True, pool_size=5, max_overflow=10)


#@st.cache_data(ttl=300, show_spinner=False)
def _sql(query: str, engine) -> pd.DataFrame:
    """Exécute une requête SQL brute via SQLAlchemy 2.x (text() obligatoire)."""
    """Cache 5 minutes."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)
