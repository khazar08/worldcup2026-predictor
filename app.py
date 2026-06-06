from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_matches
from src.model import load as load_model, feature_importances
from src.features import FEATURE_COLS
from src.predict import (
    get_all_teams,
    get_head_to_head,
    get_recent_form,
    predict_match,
)

st.set_page_config(
    page_title="2026 World Cup Predictor",
    page_icon="⚽",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    return load_model()


@st.cache_data(show_spinner="Loading match history…")
def _load_matches():
    return load_matches()


try:
    clf, le, team_states = _load_model()
    all_teams = get_all_teams(team_states)
except FileNotFoundError:
    st.error("Model artefacts not found. Run `python train.py` first, then reload.")
    st.stop()

matches = _load_matches()

st.title("⚽ 2026 FIFA World Cup Predictor")
st.caption(
    "Predictions based on ELO ratings + recent form trained on ~30 years of international results."
)


with st.sidebar:
    st.header("Match Setup")

    default1 = all_teams.index("Brazil") if "Brazil" in all_teams else 0
    default2 = all_teams.index("Argentina") if "Argentina" in all_teams else 1

    team1 = st.selectbox("Team 1 (home / left)", all_teams, index=default1)
    team2 = st.selectbox("Team 2 (away / right)", all_teams, index=default2)
    is_neutral = st.checkbox("Neutral venue", value=True)

    st.markdown("---")
    st.info(
        "All 2026 WC matches are played at neutral venues "
        "(USA / Canada / Mexico). Keep this checked."
    )

if team1 == team2:
    st.warning("Select two different teams.")
    st.stop()

result = predict_match(team1, team2, team_states, clf, le, is_neutral)
hw, d, aw = result["home_win"], result["draw"], result["away_win"]

# ELO metrics
col_t1, col_vs, col_t2 = st.columns([5, 1, 5])
with col_t1:
    st.subheader(team1)
    st.metric("ELO rating", f"{result['home_elo']:.0f}")
with col_vs:
    st.markdown("<br><br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
with col_t2:
    st.subheader(team2)
    st.metric("ELO rating", f"{result['away_elo']:.0f}")

st.markdown("---")

# Probability bar chart
st.subheader("Predicted probabilities")

fig, ax = plt.subplots(figsize=(8, 2.2))
labels = [f"{team1}\nwin", "Draw", f"{team2}\nwin"]
values = [hw, d, aw]
colors = ["#2166ac", "#878787", "#d6604d"]

bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="white")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
for bar, val in zip(bars, values):
    ax.text(
        min(bar.get_width() + 0.015, 0.95),
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1%}",
        va="center",
        fontweight="bold",
        fontsize=11,
    )
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Verdict
most_likely_label = (
    f"**{team1} wins**" if hw >= d and hw >= aw
    else ("**Draw**" if d >= aw else f"**{team2} wins**")
)
most_likely_prob = max(hw, d, aw)
st.success(f"Most likely outcome: {most_likely_label} ({most_likely_prob:.1%})")


tab_form, tab_h2h, tab_model = st.tabs(["Recent Form", "Head-to-Head", "Model Info"])

with tab_form:
    _BADGE = {"W": "🟢", "D": "🟡", "L": "🔴"}

    c1, c2 = st.columns(2)
    for col, team in ((c1, team1), (c2, team2)):
        form = get_recent_form(team, team_states, n=10)
        with col:
            st.markdown(f"**{team}** — last {len(form)} competitive matches")
            badges = " ".join(_BADGE[r] for r in form)
            summary = f"{form.count('W')}W  {form.count('D')}D  {form.count('L')}L"
            st.markdown(f"{badges}  &nbsp; `{summary}`")

            avg_scored = team_states["recent"].get(team, [])
            if avg_scored:
                scored = sum(g["scored"] for g in avg_scored) / len(avg_scored)
                conceded = sum(g["conceded"] for g in avg_scored) / len(avg_scored)
                st.caption(
                    f"Avg last {len(avg_scored)} games: "
                    f"scored {scored:.1f} / conceded {conceded:.1f}"
                )

with tab_h2h:
    h2h = get_head_to_head(team1, team2, matches, n=10)
    if h2h.empty:
        st.info(f"No recorded meetings between {team1} and {team2}.")
    else:
        res_col = f"{team1} result"
        t1w = (h2h[res_col] == "W").sum()
        dr = (h2h[res_col] == "D").sum()
        t2w = (h2h[res_col] == "L").sum()
        st.markdown(
            f"**{team1}** &nbsp;{t1w}W – {dr}D – {t2w}L&nbsp; **{team2}**"
            f"  *(last {len(h2h)} meetings)*"
        )
        display = h2h.copy()
        display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)

# ── Model info ────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("### How it works")
    st.markdown(
        """
**Model**: Gradient Boosting classifier (scikit-learn) — 3-class: home win / draw / away win.

**Training data**: ~30,000 competitive international matches from 1990 to present
([source: martj42/international_results](https://github.com/martj42/international_results)).

**Features**:
| Feature | Description |
|---|---|
| ELO ratings | Dynamic ratings updated after every match using tournament-weighted K-factors |
| ELO diff | home ELO − away ELO |
| Form | Win rate (weighted W=3/D=1) over last 10 matches |
| Goals | Avg goals scored/conceded in last 10 matches |
| Win/draw rate | Raw rates over last 10 matches |
| Neutral venue | 1 for neutral, 0 for home advantage |

**K-factor weights** (higher = more impact on ratings):
World Cup 60 · Continental tournaments 45 · Qualifiers 35 · Friendlies 25

**Disclaimer**: Football is unpredictable. These are probabilistic estimates, not guarantees.
        """
    )

    st.markdown("### Feature importances")
    fi = feature_importances(clf, FEATURE_COLS).head(10)
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    fi.iloc[::-1].plot.barh(ax=ax2, color="#2166ac", edgecolor="white")
    ax2.set_xlabel("Importance")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
