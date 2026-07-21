import streamlit as st
import numpy as np
import pickle
import os

BASE = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))


def run():
    st.header("🏠 House Price Prediction")

    area = st.number_input("Area (sq ft)", 300, 10000)
    bedrooms = st.number_input("Bedrooms", 1, 10)
    age = st.number_input("House Age (years)", 0, 100)

    if st.button("Predict Price", key="house"):
        x = np.array([[area, bedrooms, age]])
        x = scaler.transform(x)
        pred = model.predict(x)[0]
        st.success(f"Estimated Price: **${pred:,.2f}**")
