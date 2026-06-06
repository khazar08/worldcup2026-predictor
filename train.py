"""Train and save the match prediction model.

Usage
-----
    python train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_matches
from src.features import build_features, FEATURE_COLS
from src.model import train, save, feature_importances


def main() -> None:
    t0 = time.time()

    print("1/4  Loading match data...")
    matches = load_matches(min_year=1990)
    print(f"     {len(matches):,} matches loaded")

    print("2/4  Building features (ELO + form)...")
    X, y, team_states = build_features(matches, competitive_only=True)
    dist = y.value_counts().to_dict()
    print(f"     {len(X):,} training samples  |  H={dist.get('H',0):,}  D={dist.get('D',0):,}  A={dist.get('A',0):,}")

    print("3/4  Training Gradient Boosting classifier...")
    clf, le = train(X, y)

    print("4/4  Saving artefacts...")
    save(clf, le, team_states)

    print(f"\nDone in {time.time() - t0:.1f}s")
    print("\nTop-5 feature importances:")
    fi = feature_importances(clf, FEATURE_COLS)
    for feat, imp in fi.head(5).items():
        print(f"  {feat:<30} {imp:.3f}")

    print("\nRun the app:  streamlit run app.py")


if __name__ == "__main__":
    main()
