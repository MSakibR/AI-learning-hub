import streamlit as st

import house_price.app_house as house
import churn.app_churn as churn
import diabetes.app_diabetes as diabetes
import iris.app_iris as iris

st.set_page_config(page_title="AI ML Dashboard", layout="wide")

st.title("🤖 Unified Machine Learning Platform")

menu = st.sidebar.radio(
    "Select a Model",
    ["House Price", "Customer Churn", "Diabetes Risk", "Iris Classification"],
)

if menu == "House Price":
    house.run()

elif menu == "Customer Churn":
    churn.run()

elif menu == "Diabetes Risk":
    diabetes.run()

else:
    iris.run()
