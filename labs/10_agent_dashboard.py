import json, os
import pandas as pd
import plotly.express as px
import streamlit as st

LOG_FILE = "agent_log.json"

# ---------- Load logs ----------
def load_logs():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["timestamp","goal","success","score","duration_sec","feedback"])
    with open(LOG_FILE) as f:
        data = json.load(f)
    return pd.DataFrame(data)

# ---------- Dashboard ----------
st.set_page_config(page_title="Agent Monitor", layout="wide")
st.title("🧠 Agent Evaluation Dashboard")
st.write("Tracking performance, timing, and feedback of your agentic system.")

df = load_logs()

if df.empty:
    st.warning("No log data found yet. Run previous labs to generate logs.")
    st.stop()

# ---------- Metrics ----------
col1, col2, col3 = st.columns(3)
col1.metric("Total Runs", len(df))
col2.metric("Success Rate", f"{df['success'].mean()*100:.1f}%")
col3.metric("Avg Duration (s)", f"{df['duration_sec'].mean():.2f}")

# ---------- Charts ----------
st.subheader("Performance Trend")
fig_score = px.line(df, x="timestamp", y="score", title="Score Trend Over Time", markers=True)
st.plotly_chart(fig_score, use_container_width=True)

st.subheader("Duration Distribution")
fig_dur = px.histogram(df, x="duration_sec", nbins=10, title="Execution Time Distribution")
st.plotly_chart(fig_dur, use_container_width=True)

# ---------- Feedback ----------
st.subheader("Recent Evaluations")
for _, row in df.tail(5).iterrows():
    st.markdown(f"**{row['timestamp']}** — *{row['goal']}*")
    st.success(row['feedback'])
    st.divider()
