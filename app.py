"""
Sydney Property Sale Price Predictor
-------------------------------------
Streamlit web app that loads the trained Random Forest pipeline (model.pkl)
and lets a user enter property characteristics to get a predicted sale price.

Run locally with:
    pip install streamlit scikit-learn pandas
    streamlit run app.py

IMPORTANT: model.pkl must sit in the SAME FOLDER as this file.
"""

import pickle
import os
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sydney Property Price Predictor", page_icon="🏠", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


@st.cache_resource
def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


bundle = load_model()
model = bundle["model"]
num_features = bundle["num_features"]
cat_features = bundle["cat_features"]
reference_date = bundle["reference_date"]
suburb_options = bundle["suburb_options"]
type_options = bundle["type_options"]

st.title("🏠 Sydney Property Sale Price Predictor")
st.write(
    "Enter a property's characteristics below to get an estimated sale price from a "
    "Random Forest model trained on 120 manually-collected sold listings from "
    "Mosman, Parramatta and Campbelltown."
)

with st.form("property_form"):
    col1, col2 = st.columns(2)
    with col1:
        suburb = st.selectbox("Suburb", suburb_options)
        property_type = st.selectbox("Property type", type_options)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2, step=1)
    with col2:
        car_spaces = st.number_input("Car spaces", min_value=0, max_value=6, value=1, step=1)
        land_size_known = st.checkbox("Land size known?", value=True)
        land_size = st.number_input("Land size (m²)", min_value=0, max_value=5000, value=500, step=10,
                                     disabled=not land_size_known)
        sale_date = st.date_input("Sale / valuation date", value=date.today())

    submitted = st.form_submit_button("Predict sale price")

if submitted:
    land_size_missing = 0 if land_size_known else 1
    land_size_value = land_size if land_size_known else 0

    row = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces_imp": car_spaces,
        "land_size_m2_imp": land_size_value,
        "land_size_missing": land_size_missing,
        "car_spaces_missing": 0,
        "total_rooms": bedrooms + bathrooms,
        "days_since_first_sale": (pd.Timestamp(sale_date) - reference_date).days,
        "sale_month": sale_date.month,
        "suburb": suburb,
        "property_type_grouped": property_type,
    }])

    if land_size_missing == 1:
        st.info("Land size not provided — using the missing-value indicator feature; "
                 "prediction relies more heavily on suburb, bedrooms and bathrooms.")

    pred = model.predict(row[num_features + cat_features])[0]
    st.success(f"### Predicted sale price: ${pred:,.0f} AUD")
    st.caption(
        "This estimate comes from a model trained on only 120 properties across three "
        "suburbs and should be treated as an indicative starting point, not a formal "
        "valuation. See the accompanying report for model limitations."
    )

st.divider()
st.caption(
    "Model: Random Forest Regressor (scikit-learn) · Training data: 120 Domain.com.au "
    "sold listings, Mosman / Parramatta / Campbelltown · For educational use only."
)
