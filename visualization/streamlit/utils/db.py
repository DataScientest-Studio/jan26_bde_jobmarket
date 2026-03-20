"""
Connexion PostgreSQL partagée entre toutes les pages.
Utilise st.cache_resource pour une connexion unique par session Streamlit.
"""
import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text


@st.cache_resource
def get_engine():
    dsn = os.getenv("JOBSTORE_DSN", "")
    return create_engine(dsn, pool_pre_ping=True, pool_size=5, max_overflow=10)

@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str) -> pd.DataFrame:
    """Exécute une requête SQL et retourne un DataFrame. Cache 5 minutes."""
    with get_engine().connect() as conn:
        return pd.read_sql_query(text(sql), conn)
