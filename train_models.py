import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/Maternal Health Risk Data Set.csv")

X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Save models (IMPORTANT)
joblib.dump(log_model, "app/models/logistic_model.pkl")
joblib.dump(dt_model, "app/models/decision_tree_model.pkl")
joblib.dump(scaler, "app/models/scaler.pkl")

print("Models trained & saved locally ✅")

