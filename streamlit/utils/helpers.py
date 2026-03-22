import pandas as pd
from config import DARK_BG, GRID_COL, TEXT_COL, GOLD


def fmt_number(n) -> str:
    """Formate un nombre en k / M."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def fmt_euro(n) -> str:
    """Formate un salaire en € avec séparateur de milliers."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    return f"{int(n):,} €".replace(",", " ")


def kpi_card(label: str, value: str, sub: str = "", color: str = GOLD) -> str:
    """Renvoie le HTML d'une carte KPI."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent" style="background:{color}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def base_layout(title: str = "") -> dict:
    """Retourne le layout Plotly sombre commun à tous les graphiques."""
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(family="DM Sans", color=TEXT_COL, size=12),
        title=dict(
            text=title,
            font=dict(color="#d4c9b0", size=14, family="DM Serif Display"),
            x=0.02,
        ),
        margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, tickcolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, tickcolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COL),
        hoverlabel=dict(bgcolor="#1e2028", bordercolor=GRID_COL, font_color="#e8e6e0"),
    )
