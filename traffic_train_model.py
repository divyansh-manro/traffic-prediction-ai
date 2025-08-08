import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === LOAD DATA ===
df = pd.read_csv("traffic_dataset_ibm.csv")  # replace with your actual dataset filename

# === ENCODE CATEGORICAL FEATURES ===
le_date = LabelEncoder()
le_time = LabelEncoder()
le_weather = LabelEncoder()

df["Date"] = le_date.fit_transform(df["Date"])
df["Time"] = le_time.fit_transform(df["Time"])
df["Weather"] = le_weather.fit_transform(df["Weather"])

# 🔍 Make sure Junction_ID is already numeric (e.g., 0,1,2...) NOT 'J1', 'J2'
if df["Junction_ID"].dtype == object:
    df["Junction_ID"] = df["Junction_ID"].str.extract("(\d+)").astype(int) - 1  # J1 → 0

# === SELECT FEATURES & LABEL ===
features = ["Junction_ID", "Date", "Time", "Vehicle_Count",
            "Average_Speed_kmph", "Weather", "Signal_State_Green", "Signal_State_Red"]
X = df[features]
y = df["Congestion_Level"]

# === SCALE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === MODEL ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === EVALUATE ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === SAVE PICKLED ARTIFACTS ===
with open("traffic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("traffic_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

label_encoders = {
    "Date": le_date,
    "Time": le_time,
    "Weather": le_weather
}

with open("traffic_label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Training complete. Files saved.")
