
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Immutable dataclasses to avoid accidental mutations and improve code clarity. 
@dataclass(frozen=True)
class RomeItem:
    code: str
    libelle: str


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class WindowStat:
    start: str
    end: str
    total: int
