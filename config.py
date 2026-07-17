"""Central configuration. Everything downstream keys off the values here.

Tweak TICKERS freely — the pipeline, analysis, and app all read this list.
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "prices.db"

# --- Basket --------------------------------------------------------------
# Tech/semis megacaps + deliberately contrasting sectors + TLT/GLD anchors.
# The anchors (long Treasuries, gold) are what make the correlation heatmap
# tell a story instead of being uniformly high. Edit as you like.
TICKERS = [
    # Tech / semis
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AMD", "AVGO",
    # Financials
    "JPM", "GS",
    # Energy
    "XOM", "CVX",
    # Staples
    "PG", "KO",
    # Healthcare
    "JNJ", "UNH",
    # Utility
    "NEE",
    # Contrast anchors
    "TLT",   # 20+yr Treasuries
    "GLD",   # gold
]

# Sector labels, used later for correlation clustering / grouping.
SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
    "META": "Tech", "NVDA": "Semis", "AMD": "Semis", "AVGO": "Semis",
    "JPM": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy",
    "PG": "Staples", "KO": "Staples",
    "JNJ": "Healthcare", "UNH": "Healthcare",
    "NEE": "Utilities",
    "TLT": "Bonds", "GLD": "Gold",
}

# --- History -------------------------------------------------------------
HISTORY_START = "2018-01-01"   # full-history start when a ticker is not yet in the DB

# --- Analysis windows ----------------------------------------------------
CORR_WINDOWS = [30, 60, 90]        # rolling-correlation lookbacks (trading days)
VOL_WINDOW = 21                    # ~1 month for rolling volatility
TRADING_DAYS_PER_YEAR = 252

# --- Fetch resilience ----------------------------------------------------
MAX_RETRIES = 4                    # per-ticker attempts before giving up
BACKOFF_BASE_SECONDS = 2.0         # exponential backoff: base * 2**attempt
REQUEST_SPACING_SECONDS = 1.0      # polite gap between tickers to avoid rate limits
