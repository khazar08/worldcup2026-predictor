"""Integration-level tests for prediction pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from src.features import build_features
from src.predict import get_all_teams, get_head_to_head, get_recent_form, predict_match


def _synthetic_matches(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["Brazil", "Argentina", "Germany", "France", "Spain", "England"]
    rows = []
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        hs, as_ = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        rows.append({
            "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=i * 7),
            "home_team": h,
            "away_team": a,
            "home_score": hs,
            "away_score": as_,
            "tournament": "Copa América",
            "neutral": True,
            "outcome": "H" if hs > as_ else ("A" if as_ > hs else "D"),
        })
    return pd.DataFrame(rows)


def _tiny_model(df: pd.DataFrame):
    """Train a minimal GBC (no CV) for testing purposes."""
    X, y, states = build_features(df, competitive_only=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42)
    clf.fit(X, y_enc)
    return clf, le, states


# ── predict_match ─────────────────────────────────────────────────────────────

def test_probabilities_sum_to_one():
    df = _synthetic_matches()
    clf, le, states = _tiny_model(df)
    result = predict_match("Brazil", "Argentina", states, clf, le)
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_probabilities_are_non_negative():
    df = _synthetic_matches()
    clf, le, states = _tiny_model(df)
    result = predict_match("Brazil", "Germany", states, clf, le)
    assert result["home_win"] >= 0
    assert result["draw"] >= 0
    assert result["away_win"] >= 0


def test_elo_returned():
    df = _synthetic_matches()
    clf, le, states = _tiny_model(df)
    result = predict_match("France", "Spain", states, clf, le)
    assert "home_elo" in result
    assert "away_elo" in result
    assert result["home_elo"] > 0


# ── get_all_teams ─────────────────────────────────────────────────────────────

def test_get_all_teams_sorted():
    df = _synthetic_matches()
    _, _, states = build_features(df, competitive_only=True)
    teams = get_all_teams(states)
    assert teams == sorted(teams)
    assert len(teams) > 0


# ── get_recent_form ───────────────────────────────────────────────────────────

def test_recent_form_valid_symbols():
    df = _synthetic_matches()
    _, _, states = build_features(df, competitive_only=True)
    form = get_recent_form("Brazil", states, n=5)
    assert all(r in ("W", "D", "L") for r in form)


def test_recent_form_respects_n():
    df = _synthetic_matches()
    _, _, states = build_features(df, competitive_only=True)
    form = get_recent_form("Brazil", states, n=3)
    assert len(form) <= 3


def test_recent_form_unknown_team_empty():
    df = _synthetic_matches()
    _, _, states = build_features(df, competitive_only=True)
    form = get_recent_form("Atlantis FC", states, n=5)
    assert form == []


# ── get_head_to_head ──────────────────────────────────────────────────────────

def test_head_to_head_no_history():
    df = _synthetic_matches()
    h2h = get_head_to_head("TeamX", "TeamY", df, n=10)
    assert h2h.empty


def test_head_to_head_correct_result():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2022-01-01"),
        "home_team": "Brazil", "away_team": "Argentina",
        "home_score": 2, "away_score": 1,
        "tournament": "Copa América", "neutral": True, "outcome": "H",
    }])
    h2h = get_head_to_head("Brazil", "Argentina", df, n=10)
    assert len(h2h) == 1
    assert h2h["Brazil result"].iloc[0] == "W"


def test_head_to_head_respects_n():
    rows = [{
        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 30),
        "home_team": "Brazil", "away_team": "Argentina",
        "home_score": 1, "away_score": 0,
        "tournament": "Friendly", "neutral": True, "outcome": "H",
    } for i in range(15)]
    df = pd.DataFrame(rows)
    h2h = get_head_to_head("Brazil", "Argentina", df, n=5)
    assert len(h2h) == 5
