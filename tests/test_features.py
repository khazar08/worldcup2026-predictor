"""Unit tests for feature engineering (ELO, form, feature matrix)."""
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLS,
    INITIAL_ELO,
    _is_competitive,
    _k_factor,
    _stats,
    build_features,
    match_features,
)


def _sample_matches() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01", "2022-01-01",
        ]),
        "home_team": ["Brazil", "Argentina", "Brazil", "Germany", "France"],
        "away_team": ["Argentina", "Brazil", "Germany", "France", "Brazil"],
        "home_score": [2, 1, 0, 3, 1],
        "away_score": [1, 1, 2, 0, 2],
        "tournament": [
            "Copa América", "Copa América", "FIFA World Cup", "UEFA Euro",
            "FIFA World Cup",
        ],
        "neutral": [True, True, True, True, True],
        "outcome": ["H", "D", "A", "H", "A"],
    })


# ── _stats ────────────────────────────────────────────────────────────────────

def test_stats_empty_returns_defaults():
    s = _stats([])
    assert s["win_rate"] == 0.33
    assert s["draw_rate"] == 0.33
    assert s["goals_scored"] == 1.0
    assert s["goals_conceded"] == 1.0


def test_stats_all_wins():
    games = [{"won": True, "drew": False, "scored": 3, "conceded": 0}] * 4
    s = _stats(games)
    assert s["win_rate"] == 1.0
    assert s["draw_rate"] == 0.0
    assert s["form"] == 1.0
    assert s["goals_scored"] == 3.0


def test_stats_all_draws():
    games = [{"won": False, "drew": True, "scored": 1, "conceded": 1}] * 4
    s = _stats(games)
    assert s["win_rate"] == 0.0
    assert s["draw_rate"] == 1.0
    assert abs(s["form"] - 1 / 3) < 1e-9


def test_stats_goals_average():
    games = [
        {"won": True, "drew": False, "scored": 2, "conceded": 0},
        {"won": False, "drew": False, "scored": 0, "conceded": 2},
    ]
    s = _stats(games)
    assert s["goals_scored"] == 1.0
    assert s["goals_conceded"] == 1.0


# ── _k_factor ─────────────────────────────────────────────────────────────────

def test_k_factor_world_cup():
    assert _k_factor("FIFA World Cup") == 60


def test_k_factor_qualifier():
    assert _k_factor("FIFA World Cup qualification") == 35


def test_k_factor_friendly_uses_default():
    assert _k_factor("International friendly") == 25


def test_k_factor_copa_america():
    assert _k_factor("Copa América") == 45


# ── _is_competitive ───────────────────────────────────────────────────────────

def test_competitive_tournament():
    assert _is_competitive("FIFA World Cup") is True
    assert _is_competitive("UEFA Euro qualification") is True


def test_friendly_not_competitive():
    assert _is_competitive("International friendly") is False
    assert _is_competitive("Friendly") is False


# ── build_features ────────────────────────────────────────────────────────────

def test_build_features_column_names():
    X, y, _ = build_features(_sample_matches(), competitive_only=True)
    assert list(X.columns) == FEATURE_COLS


def test_build_features_lengths_match():
    X, y, _ = build_features(_sample_matches(), competitive_only=True)
    assert len(X) == len(y)


def test_build_features_valid_outcomes():
    _, y, _ = build_features(_sample_matches(), competitive_only=True)
    assert set(y.unique()).issubset({"H", "D", "A"})


def test_build_features_elo_changes():
    _, _, states = build_features(_sample_matches(), competitive_only=True)
    assert states["elo"]["Brazil"] != INITIAL_ELO
    assert states["elo"]["Argentina"] != INITIAL_ELO


def test_build_features_recent_games_populated():
    _, _, states = build_features(_sample_matches(), competitive_only=True)
    assert len(states["recent"]["Brazil"]) > 0


# ── match_features ────────────────────────────────────────────────────────────

def test_match_features_shape():
    _, _, states = build_features(_sample_matches(), competitive_only=True)
    X = match_features("Brazil", "Argentina", states, is_neutral=True)
    assert X.shape == (1, len(FEATURE_COLS))
    assert list(X.columns) == FEATURE_COLS


def test_match_features_neutral_flag():
    _, _, states = build_features(_sample_matches(), competitive_only=True)
    Xn = match_features("Brazil", "Argentina", states, is_neutral=True)
    Xh = match_features("Brazil", "Argentina", states, is_neutral=False)
    assert Xn["is_neutral"].iloc[0] == 1
    assert Xh["is_neutral"].iloc[0] == 0


def test_match_features_unknown_team_uses_default_elo():
    _, _, states = build_features(_sample_matches(), competitive_only=True)
    X = match_features("Atlantis FC", "Brazil", states, is_neutral=True)
    assert X["home_elo"].iloc[0] == INITIAL_ELO
