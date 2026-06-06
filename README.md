# 2026 FIFA World Cup Predictor

Predict **win / draw / loss** probabilities for any international football match using a Gradient Boosting model trained on 35 years of FIFA results.

## Features

- **ELO rating system** — dynamic ratings updated after every match with tournament-weighted K-factors (World Cup = 60, friendlies = 25)
- **Rolling team form** — win rate, goals scored/conceded in last 10 competitive matches
- **Gradient Boosting classifier** — 3-class (H / D / A), ~87% CV accuracy on held-out competitive matches
- **Interactive Streamlit app** — pick any two nations, see probabilities + head-to-head history

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/worldcup2026-predictor
cd worldcup2026-predictor

pip install -r requirements.txt

# Download data (auto), build features, train model (~60s)
python3 train.py

# Launch the app
streamlit run app.py
```

## Project structure

```
worldcup2026-predictor/
├── app.py              # Streamlit UI
├── train.py            # Feature engineering + model training
├── requirements.txt
├── src/
│   ├── data_loader.py  # Download & parse results.csv
│   ├── features.py     # ELO + rolling form → feature matrix
│   ├── model.py        # Train / save / load GradientBoosting
│   └── predict.py      # Match prediction & lookup helpers
├── data/               # results.csv downloaded here (gitignored)
└── models/             # model.pkl + team_states.pkl (gitignored)
```

## Data source

[martj42/international_results](https://github.com/martj42/international_results) — 50,000+ international matches from 1872 to present. Downloaded automatically on first run.

## How predictions work

1. **ELO ratings** are computed iteratively across all historical matches (ELO diff is the strongest predictor).
2. **Rolling stats** (form, goals) are tracked per team in a 10-game window.
3. At prediction time, the model receives the current ELO and form of both teams as a 14-feature vector.
4. All 2026 WC matches use `is_neutral=1` (hosted in USA/Canada/Mexico).

## Model performance

Typical 5-fold cross-validated accuracy: **~54–56%** on competitive matches.  
(Baseline: always predict home win ≈ 45%. Predicting football is hard.)

<img width="1512" height="809" alt="Screenshot 2026-06-07 at 12 11 26 AM" src="https://github.com/user-attachments/assets/f564daed-cdc4-4d21-a592-ea2c6096bd08" />
<img width="1510" height="813" alt="Screenshot 2026-06-07 at 12 13 22 AM" src="https://github.com/user-attachments/assets/90be757a-2995-4b5e-83ff-94508a23f79e" />
<img width="1512" height="708" alt="Screenshot 2026-06-07 at 12 13 08 AM" src="https://github.com/user-attachments/assets/458e7244-9eb0-48b7-85aa-3d9a7f9cecc2" />


## Limitations

- ELO assumes a stationary rating system — team quality changes faster than the model adapts after injuries or coaching changes.
- Training on all competitive matches equally weights a 2010 qualifier the same as a 2024 Nations League final. Recency weighting would help.
- The model has no knowledge of squad depth, injuries, suspensions, or weather.
- Draw probability is the hardest class — the model tends to underweight it.
- Small-nation matches are underrepresented in training data.



## Roadmap

Add FIFA official rankings as an additional feature alongside ELO
Weight recent matches more heavily than older ones (time-decay on training samples)
Add a group stage simulator — run N Monte Carlo simulations of the full group stage and show qualification probabilities for all 48 teams
Add a knockout bracket simulator through to the final
Display team flags in the UI
Add a "biggest upsets" view showing matches where the model's underdog actually won
Experiment with XGBoost or a neural network to see if draw prediction improves

