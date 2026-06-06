"""High-level prediction and lookup helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from src.features import match_features, INITIAL_ELO
from src.model import predict_proba


def get_all_teams(team_states: dict[str, Any]) -> list[str]:
    """Sorted list of all teams with known ELO ratings."""
    return sorted(team_states["elo"].keys())


def predict_match(
    home_team: str,
    away_team: str,
    team_states: dict[str, Any],
    clf: GradientBoostingClassifier,
    le: LabelEncoder,
    is_neutral: bool = True,
) -> dict[str, Any]:
    """
    Predict outcome probabilities for a match.

    Returns
    -------
    dict with keys: home_win, draw, away_win (floats), home_elo, away_elo
    """
    X = match_features(home_team, away_team, team_states, is_neutral)
    probs = predict_proba(clf, le, X)
    return {
        "home_win": probs.get("H", 0.0),
        "draw": probs.get("D", 0.0),
        "away_win": probs.get("A", 0.0),
        "home_elo": round(team_states["elo"].get(home_team, INITIAL_ELO), 1),
        "away_elo": round(team_states["elo"].get(away_team, INITIAL_ELO), 1),
    }


def get_recent_form(team: str, team_states: dict[str, Any], n: int = 10) -> list[str]:
    """Last n results oldest→newest as 'W'/'D'/'L'."""
    games = team_states["recent"].get(team, [])[-n:]
    out = []
    for g in games:
        if g["won"]:
            out.append("W")
        elif g["drew"]:
            out.append("D")
        else:
            out.append("L")
    return out


def get_head_to_head(
    team1: str, team2: str, matches: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """Last n encounters between two teams, most-recent first."""
    mask = (
        ((matches["home_team"] == team1) & (matches["away_team"] == team2))
        | ((matches["home_team"] == team2) & (matches["away_team"] == team1))
    )
    h2h = matches[mask].tail(n).copy()
    if h2h.empty:
        return h2h

    def _result(row: pd.Series) -> str:
        if row["home_team"] == team1:
            return {"H": "W", "D": "D", "A": "L"}[row["outcome"]]
        return {"A": "W", "D": "D", "H": "L"}[row["outcome"]]

    h2h[f"{team1} result"] = h2h.apply(_result, axis=1)
    cols = ["date", "home_team", "home_score", "away_score", "away_team",
            "tournament", f"{team1} result"]
    return h2h[cols].iloc[::-1].reset_index(drop=True)
