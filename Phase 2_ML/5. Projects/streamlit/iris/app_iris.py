import streamlit as st
import numpy as np
import pickle
import os

BASE = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE, "scaler.pkl"), "rb"))
labels = pickle.load(open(os.path.join(BASE, "labels.pkl"), "rb"))


def run():
    st.header("🌸 Iris Flower Classification")

    sl = st.number_input("Sepal Length", 0.0, 10.0)
    sw = st.number_input("Sepal Width", 0.0, 10.0)
    pl = st.number_input("Petal Length", 0.0, 10.0)
    pw = st.number_input("Petal Width", 0.0, 10.0)

    if st.button("Predict Flower", key="iris"):
        x = np.array([[sl, sw, pl, pw]])
        x = scaler.transform(x)
        pred = model.predict(x)[0]

        st.success(f"🌼 Flower Type: **{labels[pred]}**")
