import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle, os

data = pd.read_csv("diabetes.csv")

X = data.drop("outcome", axis=1)
y = data["outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Diabetes Model Saved Successfully 🎯")
