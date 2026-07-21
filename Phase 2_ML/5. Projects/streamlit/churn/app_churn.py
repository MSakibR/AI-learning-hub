import streamlit as st
import numpy as np
import pickle
import os

BASE = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))


def run():
    st.header("📞 Customer Churn Prediction")

    age = st.number_input("Age", 18, 100)
    salary = st.number_input("Salary", 10000, 300000)
    balance = st.number_input("Account Balance", 0, 500000)
    products = st.number_input("Products Owned", 1, 4)
    active = st.selectbox("Active Member?", [1, 0])

    if st.button("Predict Churn", key="churn"):
        x = np.array([[age, salary, balance, products, active]])
        x = scaler.transform(x)
        pred = model.predict(x)[0]

        st.warning("❌ Will Churn") if pred == 1 else st.success("✅ Will Stay")
