from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Absolute path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://maternity-ml-frontend.vercel.app"],  # for testing, or use your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    # MaternityMonth is accepted but not used by the current model
    MaternityMonth: int = 0

@app.post("/predict/logistic")
def predict_logistic(data: PatientData):
    try:
        # The model was trained with 6 features: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
        X = np.array([[data.Age, data.SystolicBP, data.DiastolicBP,
                       data.BS, data.BodyTemp, data.HeartRate]])
        
        X_scaled = scaler.transform(X)
        pred = log_model.predict(X_scaled)[0]
        
        # Binary or Multi-class? The dataset has 'low risk', 'mid risk', 'high risk'
        # LogisticRegression might return string or int depending on label encoding
        return {"RiskLevel": str(pred)}
    except Exception as e:
        print(f"Error in /predict/logistic: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/tree")
def predict_tree(data: PatientData):
    try:
        # The model was trained with 6 features: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
        X = np.array([[data.Age, data.SystolicBP, data.DiastolicBP,
                       data.BS, data.BodyTemp, data.HeartRate]])
        
        pred = dt_model.predict(X)[0]
        return {"RiskLevel": str(pred)}
    except Exception as e:
        print(f"Error in /predict/tree: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Maternity Risk Prediction API is running"}


