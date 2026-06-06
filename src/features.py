"""Feature engineering: ELO ratings + rolling team form."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import pandas as pd

INITIAL_ELO = 1500.0
N_RECENT = 10

# Higher K = bigger rating swings for more important tournaments.
# "qualification" must come before tournament names so "FIFA World Cup
# qualification" gets K=35, not K=60.
_K_MAP = {
    "qualification": 35,
    "fifa world cup": 60,
    "confederations cup": 50,
    "uefa nations league": 45,
    "copa am": 45,  # Copa América
    "uefa euro": 45,
    "africa cup": 45,
    "gold cup": 40,
    "asian cup": 40,
}
_K_DEFAULT = 25  # friendlies / minor tournaments

FEATURE_COLS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form",
    "away_form",
    "home_goals_scored",
    "away_goals_scored",
    "home_goals_conceded",
    "away_goals_conceded",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "is_neutral",
]


def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    for key, k in _K_MAP.items():
        if key in t:
            return float(k)
    return float(_K_DEFAULT)


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _stats(games: list[dict]) -> dict[str, float]:
    if not games:
        return dict(form=0.33, goals_scored=1.0, goals_conceded=1.0,
                    win_rate=0.33, draw_rate=0.33)
    n = len(games)
    wins = sum(1 for g in games if g["won"])
    draws = sum(1 for g in games if g["drew"])
    return dict(
        form=(wins * 3 + draws) / (n * 3),
        goals_scored=sum(g["scored"] for g in games) / n,
        goals_conceded=sum(g["conceded"] for g in games) / n,
        win_rate=wins / n,
        draw_rate=draws / n,
    )


def _is_competitive(tournament: str) -> bool:
    return "friendly" not in tournament.lower()


def build_features(
    matches: pd.DataFrame,
    competitive_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Single-pass feature builder. Iterates matches chronologically,
    tracking ELO and rolling form for every team.

    Parameters
    ----------
    matches : sorted by date DataFrame from data_loader.load_matches()
    competitive_only : if True, only include non-friendly matches in X/y

    Returns
    -------
    X : feature DataFrame (FEATURE_COLS columns)
    y : outcome Series ('H', 'D', 'A')
    team_states : dict with 'elo' and 'recent' for live inference
    """
    elo: dict[str, float] = defaultdict(lambda: INITIAL_ELO)
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=N_RECENT))

    rows: list[dict] = []
    labels: list[str] = []

    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        h_elo, a_elo = elo[h], elo[a]
        h_st = _stats(list(recent[h]))
        a_st = _stats(list(recent[a]))

        if not competitive_only or _is_competitive(row["tournament"]):
            feat: dict[str, float] = {
                "home_elo": h_elo,
                "away_elo": a_elo,
                "elo_diff": h_elo - a_elo,
                "home_form": h_st["form"],
                "away_form": a_st["form"],
                "home_goals_scored": h_st["goals_scored"],
                "away_goals_scored": a_st["goals_scored"],
                "home_goals_conceded": h_st["goals_conceded"],
                "away_goals_conceded": a_st["goals_conceded"],
                "home_win_rate": h_st["win_rate"],
                "away_win_rate": a_st["win_rate"],
                "home_draw_rate": h_st["draw_rate"],
                "away_draw_rate": a_st["draw_rate"],
                "is_neutral": int(bool(row["neutral"])),
            }
            rows.append(feat)
            labels.append(row["outcome"])

        # Update ELO for ALL matches (friendlies count, just with lower K)
        outcome = row["outcome"]
        sh = 1.0 if outcome == "H" else (0.5 if outcome == "D" else 0.0)
        sa = 0.5 if outcome == "D" else 1.0 - sh
        k = _k_factor(str(row["tournament"]))
        exp_h = _expected(h_elo, a_elo)
        elo[h] = h_elo + k * (sh - exp_h)
        elo[a] = a_elo + k * (sa - (1.0 - exp_h))

        recent[h].append(
            dict(scored=row["home_score"], conceded=row["away_score"],
                 won=outcome == "H", drew=outcome == "D")
        )
        recent[a].append(
            dict(scored=row["away_score"], conceded=row["home_score"],
                 won=outcome == "A", drew=outcome == "D")
        )

    team_states: dict[str, Any] = {
        "elo": dict(elo),
        "recent": {t: list(q) for t, q in recent.items()},
    }
    return (
        pd.DataFrame(rows, columns=FEATURE_COLS),
        pd.Series(labels, name="outcome"),
        team_states,
    )


def match_features(
    home_team: str,
    away_team: str,
    team_states: dict[str, Any],
    is_neutral: bool = True,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for a future match."""
    h_elo = team_states["elo"].get(home_team, INITIAL_ELO)
    a_elo = team_states["elo"].get(away_team, INITIAL_ELO)
    h_st = _stats(team_states["recent"].get(home_team, []))
    a_st = _stats(team_states["recent"].get(away_team, []))

    row: dict[str, float] = {
        "home_elo": h_elo,
        "away_elo": a_elo,
        "elo_diff": h_elo - a_elo,
        "home_form": h_st["form"],
        "away_form": a_st["form"],
        "home_goals_scored": h_st["goals_scored"],
        "away_goals_scored": a_st["goals_scored"],
        "home_goals_conceded": h_st["goals_conceded"],
        "away_goals_conceded": a_st["goals_conceded"],
        "home_win_rate": h_st["win_rate"],
        "away_win_rate": a_st["win_rate"],
        "home_draw_rate": h_st["draw_rate"],
        "away_draw_rate": a_st["draw_rate"],
        "is_neutral": int(is_neutral),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)
