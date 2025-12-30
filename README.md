# Maternity Risk Prediction API

This project provides an API to predict maternal health risk levels (low, mid, high) based on various health parameters using Machine Learning models.

## 🚀 Features
- **Logistic Regression Model** for risk prediction
- **Decision Tree Model** for risk prediction
- Built with **FastAPI** for high performance

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/melody2203/maternity-ml-backend.git
   cd maternity-ml-backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the API

Start the server using Uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## 📡 API Endpoints

### 1️⃣ Health Check
- **GET** `/`
- Returns: `{"message": "Maternity Risk Prediction API is running"}`

### 2️⃣ Predict Risk (Logistic Regression)
- **POST** `/predict/logistic`
- **Body:**
  ```json
  {
    "Age": 25,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 7.0,
    "BodyTemp": 98.0,
    "HeartRate": 70,
    "MaternityMonth": 3
  }
  ```
- **Response:** `{"RiskLevel": <int>}`

### 3️⃣ Predict Risk (Decision Tree)
- **POST** `/predict/tree`
- **Body:** Same as above
- **Response:** `{"RiskLevel": <int>}`

## 📂 Project Structure
- `app/main.py`: Main API application
- `train_models.py`: Script to train and save ML models
- `data/`: Dataset directory
- `app/models/`: Directory where trained models are saved
