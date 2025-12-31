import requests

url = "http://127.0.0.1:8000/predict/tree"
data = {
    "Age": 25,
    "SystolicBP": 120,
    "DiastolicBP": 80,
    "BS": 7.0,
    "BodyTemp": 98.0,
    "HeartRate": 70,
    "MaternityMonth": 3
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
