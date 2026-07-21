import streamlit as st
import numpy as np
import pickle
import os

BASE = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))


def run():
    st.header("🩺 Diabetes Prediction")

    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 300)
    bp = st.number_input("Blood Pressure", 0, 200)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes", key="diabetes"):
        x = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        x = scaler.transform(x)
        pred = model.predict(x)[0]

        st.error("⚠️ High Risk") if pred == 1 else st.success("✅ Low Risk")
