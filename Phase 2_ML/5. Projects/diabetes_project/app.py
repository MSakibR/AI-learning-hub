import pickle
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE, "model", "diabetes_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE, "model", "scaler.pkl"), "rb"))

print("\n----- DIABETES PREDICTOR -----\n")

vals = []
fields = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree",
    "Age",
]

for f in fields:
    vals.append(float(input(f"{f}: ")))

data = np.array([vals])
data = scaler.transform(data)

pred = model.predict(data)[0]

print("\nResult:")
print("High Risk of Diabetes ⚠️" if pred == 1 else "Low Risk of Diabetes ✅")
