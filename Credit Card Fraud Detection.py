# Preprocessing & Model Training
# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load data
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Address class imbalance
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Save artifacts
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "model.pkl")


#Real‑Time API Endpoint (FastAPI)
# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
scaler = joblib.load("../scaler.pkl")
model = joblib.load("../model.pkl")

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.post("/predict/")
def predict(tx: Transaction):
    df = pd.DataFrame([tx.dict()])
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = float(model.predict_proba(scaled)[0][1])
    return {"is_fraud": int(pred), "fraud_prob": prob}


#Real-Time Dashboard (Streamlit)
# dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import random

API = "http://127.0.0.1:8000/predict/"

st.title("Real‑Time Fraud Detection")

if st.button("Simulate Transaction"):
    tx = {
        "Time": random.uniform(0, 100000),
        **{f"V{i}": random.uniform(-5,5) for i in range(1,29)},
        "Amount": random.uniform(0,500)
    }
    resp = requests.post(API, json=tx).json()
    st.json(resp)
    st.write("⚠️ Fraud!" if resp["is_fraud"] else "✅ OK")


#Run via:
streamlit run dashboard/app.py


