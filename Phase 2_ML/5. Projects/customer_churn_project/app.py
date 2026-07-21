import pickle
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE, "model", "churn_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE, "model", "scaler.pkl"), "rb"))

print("\n----- CUSTOMER CHURN PREDICTOR -----\n")

age = float(input("Age: "))
salary = float(input("Salary: "))
balance = float(input("Account Balance: "))
products = float(input("Number of Products: "))
active = float(input("Active Member? (1=yes, 0=no): "))

data = np.array([[age, salary, balance, products, active]])
data = scaler.transform(data)

pred = model.predict(data)[0]

print("\nResult:")
print("Customer WILL CHURN ❌" if pred == 1 else "Customer WILL STAY ✅")
