from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Absolute path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

log_model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
dt_model = joblib.load(os.path.join(MODEL_DIR, "decision_tree_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

class PatientData(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float
    MaternityMonth: int

@app.post("/predict/logistic")
def predict_logistic(data: PatientData):
    X = np.array([[data.Age, data.SystolicBP, data.DiastolicBP,
                   data.BS, data.BodyTemp, data.HeartRate,
                   data.MaternityMonth]])
    X_scaled = scaler.transform(X)
    pred = log_model.predict(X_scaled)[0]
    return {"RiskLevel": int(pred)}

@app.post("/predict/tree")
def predict_tree(data: PatientData):
    X = np.array([[data.Age, data.SystolicBP, data.DiastolicBP,
                   data.BS, data.BodyTemp, data.HeartRate,
                   data.MaternityMonth]])
    pred = dt_model.predict(X)[0]
    return {"RiskLevel": int(pred)}
@app.get("/")
def root():
    return {"message": "Maternity Risk Prediction API is running"}


