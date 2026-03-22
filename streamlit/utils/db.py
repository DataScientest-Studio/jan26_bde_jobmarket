import streamlit as st
from sqlalchemy import create_engine
from config import DB_URL


@st.cache_resource
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)
