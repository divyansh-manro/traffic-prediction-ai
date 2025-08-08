# Traffic Congestion Prediction & Smart Agent System

This project presents an AI-powered traffic congestion prediction system integrated with a rule-based traffic agent. Built using **Python, Streamlit, scikit-learn**, and **Google Gemini API**, and deployed on **IBM Cloud**, the app enables real-time congestion forecasting and dynamic traffic signal control.

---

## 🚦 Features

- Predicts traffic congestion using trained ML models.
- Real-time user inputs: junction, time, vehicle count, weather.
- Suggests traffic control decisions via a rule-based agent.
- Utilizes Google Gemini for explanation and summarization.
- Streamlit-powered user interface.
- IBM Cloud ready for scalable deployment.

---

## 🧠 Technologies Used

- Python 3.11
- scikit-learn
- Streamlit
- Google Gemini API
- IBM Cloud
- TensorFlow / Pickle for model handling

---

## 📁 Project Structure

```bash
traffic-prediction-ai/
├── traffic_ai_with_agent.py     # Main app logic (Streamlit interface)
├── traffic_model.pkl            # Trained ML model (pickled)
├── label_encoder.pkl            # Encoded labels for junctions
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
