"""Load and preprocess international football match data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
DATA_PATH = Path(__file__).parent.parent / "data" / "results.csv"


def download_data(force: bool = False) -> None:
    """Download match history from GitHub if not already cached."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return
    print(f"Downloading data from GitHub...")
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    DATA_PATH.write_bytes(r.content)
    print(f"Saved {len(r.content) / 1024:.0f} KB → {DATA_PATH}")


def load_matches(min_year: int = 1990) -> pd.DataFrame:
    """
    Load international match results from 1990 onward.

    Returns
    -------
    DataFrame with columns: date, home_team, away_team, home_score,
    away_score, tournament, neutral, outcome ('H'/'D'/'A')
    """
    download_data()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["date"].dt.year >= min_year].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["outcome"] = "D"
    df.loc[df["home_score"] > df["away_score"], "outcome"] = "H"
    df.loc[df["home_score"] < df["away_score"], "outcome"] = "A"

    return df.sort_values("date").reset_index(drop=True)
