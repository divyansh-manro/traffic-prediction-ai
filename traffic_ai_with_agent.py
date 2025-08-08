import streamlit as st
import numpy as np
import pickle
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Traffic Congestion Predictor & AI Agent", layout="wide")
st.title("🚦 Prediction & Agent Simulation")
st.markdown("This app predicts congestion and lets an agent take action, assisted by **Google Gemini**.")

# === LOAD MODEL & SCALER ===
with open("traffic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("traffic_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("traffic_label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

# Unpack individual encoders
le_date = le_dict['Date']
le_time = le_dict['Time']
le_weather = le_dict['Weather']

# === SIDEBAR USER INPUTS ===
st.sidebar.header("Input Traffic Data")

# ✅ Use encoded form directly (e.g., 0 for J1, 1 for J2)
junction = st.sidebar.selectbox("Junction ID", [0, 1], format_func=lambda x: f"J{x+1}")
date = st.sidebar.selectbox("Date", le_date.classes_.tolist())
time = st.sidebar.selectbox("Time", le_time.classes_.tolist())
vehicle_count = st.sidebar.slider("Vehicle Count", 0, 200, 50)
speed = st.sidebar.slider("Average Speed (km/h)", 0, 120, 45)
weather = st.sidebar.selectbox("Weather", le_weather.classes_.tolist())
green = st.sidebar.slider("Signal State - Green (sec)", 0, 120, 50)
red = st.sidebar.slider("Signal State - Red (sec)", 0, 120, 50)

# === ENCODE CATEGORICAL FEATURES ===
date_encoded = le_date.transform([date])[0]
time_encoded = le_time.transform([time])[0]
weather_encoded = le_weather.transform([weather])[0]

# === PREPARE FINAL INPUT ===
input_features = np.array([[junction, date_encoded, time_encoded, vehicle_count,
                            speed, weather_encoded, green, red]])
input_scaled = scaler.transform(input_features)

# === PREDICT ===
prediction = model.predict(input_scaled)[0]
congestion_map = {0: "Low", 1: "Medium", 2: "High"}
congestion_level = congestion_map.get(prediction, "Unknown")

st.subheader("🚗 Predicted Congestion Level")
st.success(f"**{congestion_level}** congestion expected at Junction {junction + 1}")

# === GEMINI AGENT INTERACTION ===
st.subheader("🤖 Agent Suggestion")

# Google Gemini Setup
genai.configure(api_key="Your_Key_Here")  # Replace with your key
model_gemini = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""
You are a traffic management AI agent.

Given the current traffic scenario:
- Junction: J{junction + 1}
- Date: {date}
- Time: {time}
- Vehicle Count: {vehicle_count}
- Speed: {speed} km/h
- Weather: {weather}
- Signal: Green={green}s, Red={red}s
- Predicted Congestion: {congestion_level}

Suggest actions like:
- Adjusting signal timings
- Rerouting traffic
- Alerting nearby junctions
- Triggering emergency responses (if needed)

Provide a short and precise recommendation.
"""

if st.button("💡 Ask Agent"):
    with st.spinner("Consulting agent..."):
        response = model_gemini.generate_content(prompt)
        st.info(response.text)

# === FOOTER ===
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, XGBoost, and Gemini AI")
