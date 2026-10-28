"""Streamlit demo for the personal digital twin."""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__:
    from .data_sim import simulate_data
    from .explain import feature_importance
    from .model_train import train_model
    from .predict import predict_wellbeing
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_sim import simulate_data
    from src.explain import feature_importance
    from src.model_train import train_model
    from src.predict import predict_wellbeing


@st.cache_data
def load_data():
    return simulate_data()


@st.cache_resource
def load_model():
    return train_model(load_data())[0]


def recommendations(observation: dict[str, float], prediction: float) -> list[str]:
    """Create short, personalized actions from the user's current inputs."""
    actions = []
    if observation["sleep_hours"] < 7:
        actions.append("Protect an earlier wind-down tonight: sleep is your biggest immediate lever.")
    if observation["exercise_minutes"] < 30:
        actions.append("Try a 20-minute walk or stretch break to build movement into the day.")
    if observation["screen_hours"] > 7:
        actions.append("Set one screen-free block this evening, especially during your wind-down.")
    if observation["focus_hours"] < 4:
        actions.append("Choose one small priority and schedule a distraction-free focus block.")
    if observation["social_minutes"] < 30:
        actions.append("Add a brief check-in with someone who helps you feel connected.")
    if not actions:
        actions.append("Your inputs are balanced today. Keep the routine steady and notice what helps.")
    if prediction >= 80:
        actions.insert(0, "Your projected day is strong. Preserve the habits that are working.")
    return actions[:3]


def get_history() -> pd.DataFrame:
    """Return the user's saved observations from the current session."""
    return pd.DataFrame(st.session_state.get("history", []))


st.set_page_config(page_title="Personal Digital Twin", page_icon="🧭", layout="wide")
st.title("Personal Digital Twin")
st.caption("A private, session-based wellbeing companion that learns from your daily check-ins.")

if "history" not in st.session_state:
    st.session_state.history = []

model = load_model()
with st.sidebar:
    st.header("Your profile")
    profile_name = st.text_input("Name", value=st.session_state.get("profile_name", "Friend"))
    goal = st.selectbox(
        "Today's focus",
        ["Balanced energy", "Better sleep", "Deeper focus", "More movement", "More connection"],
    )
    st.session_state.profile_name = profile_name
    st.divider()
    st.header("Today's check-in")
    selected_date = st.date_input("Date", value=date.today())
    observation = {
        "sleep_hours": st.slider("Sleep (hours)", 4.0, 10.0, 7.0),
        "exercise_minutes": st.slider("Exercise (minutes)", 0.0, 150.0, 30.0),
        "focus_hours": st.slider("Focus (hours)", 0.0, 10.0, 5.0),
        "screen_hours": st.slider("Screen time (hours)", 0.0, 12.0, 5.0),
        "social_minutes": st.slider("Social time (minutes)", 0.0, 240.0, 60.0),
    }
    save_checkin = st.button("Save today's check-in", type="primary", use_container_width=True)
    if save_checkin:
        entry = {"date": str(selected_date), **observation}
        entry["wellbeing"] = predict_wellbeing(model, observation)
        st.session_state.history = [
            item for item in st.session_state.history if item["date"] != entry["date"]
        ] + [entry]
        st.success("Check-in saved")

prediction = predict_wellbeing(model, observation)
history = get_history()
previous = history.iloc[-2]["wellbeing"] if len(history) > 1 else None

left, middle, right = st.columns(3)
left.metric("Live wellbeing", f"{prediction:.1f} / 100", None if previous is None else f"{prediction - previous:+.1f} vs last")
middle.metric("Today's focus", goal)
middle.caption(f"A check-in tailored for {profile_name or 'you'}")
right.metric("Saved check-ins", len(history))

st.subheader(f"{profile_name or 'Your'} daily readout")
for action in recommendations(observation, prediction):
    st.info(action)

if not history.empty:
    st.subheader("Your trend")
    chart_data = history.copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"])
    st.line_chart(chart_data.set_index("date")["wellbeing"])
    st.dataframe(
        history.sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download my check-ins",
        history.to_csv(index=False).encode("utf-8"),
        file_name="personal-digital-twin-checkins.csv",
        mime="text/csv",
    )
else:
    st.warning("Save your first check-in to start building your personal trend.")

with st.expander("Model signals"):
    st.bar_chart(feature_importance(model))
    st.caption("This uses synthetic training data and your session history. It is not medical advice.")

if history.empty is False and st.button("Clear saved check-ins"):
    st.session_state.history = []
    st.rerun()
