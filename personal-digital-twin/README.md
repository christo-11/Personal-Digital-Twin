# Personal Digital Twin

A small Python project for experimenting with synthetic personal behavior data, wellbeing prediction, causal summaries, and model explanations.

## Setup

```powershell
cd personal-digital-twin
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Run the tests:

```powershell
pytest
```

Launch the personalized dashboard:

```powershell
python -m streamlit run src/app.py
```

The dashboard recalculates wellbeing as you change today's inputs, lets you save daily check-ins in the current session, shows your trend, generates tailored suggestions, and exports your check-ins as CSV. The model is trained on synthetic data only; this is an educational prototype and is not medical advice.
