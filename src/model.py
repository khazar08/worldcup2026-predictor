"""Model training, saving, and loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = _PROJECT_ROOT / "models" / "model.pkl"
STATES_PATH = _PROJECT_ROOT / "models" / "team_states.pkl"


def train(
    X: pd.DataFrame, y: pd.Series
) -> tuple[GradientBoostingClassifier, LabelEncoder]:
    """
    Train a gradient boosting classifier and report cross-validated accuracy.

    Returns
    -------
    clf : fitted classifier
    le  : LabelEncoder mapping H/D/A → ints
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
    print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y_enc)
    return clf, le


def save(
    clf: GradientBoostingClassifier,
    le: LabelEncoder,
    team_states: dict[str, Any],
) -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"clf": clf, "le": le}, MODEL_PATH)
    joblib.dump(team_states, STATES_PATH)
    print(f"  Saved model    → {MODEL_PATH}")
    print(f"  Saved states   → {STATES_PATH}")


def load() -> tuple[GradientBoostingClassifier, LabelEncoder, dict[str, Any]]:
    """Load saved model, label encoder, and team states."""
    data = joblib.load(MODEL_PATH)
    team_states = joblib.load(STATES_PATH)
    return data["clf"], data["le"], team_states


def predict_proba(
    clf: GradientBoostingClassifier,
    le: LabelEncoder,
    X: pd.DataFrame,
) -> dict[str, float]:
    """Return {outcome_label: probability} for a single-row feature DataFrame."""
    probs = clf.predict_proba(X)[0]
    return dict(zip(le.classes_, probs))


def feature_importances(
    clf: GradientBoostingClassifier, feature_names: list[str]
) -> pd.Series:
    return (
        pd.Series(clf.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )
