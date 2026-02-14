import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# ----------------------------
# Load Model & Preprocessors
# ----------------------------

model = tf.keras.models.load_model("ann_model.h5")

with open("one_hot_encoder_geography.pkl", "rb") as f:
    one_hot_encoder_geography = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# UI
# ----------------------------

st.title("Customer Churn Prediction")

credit_score = st.slider("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict"):

    # Raw Input
    input_data = {
        "CreditScore": credit_score,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
        "Geography": geography,
    }

    input_df = pd.DataFrame([input_data])

    # ----------------------------
    # Encode Gender
    # ----------------------------
    input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])

    # ----------------------------
    # Encode Geography (SAFE VERSION)
    # ----------------------------
    geo_encoded = one_hot_encoder_geography.transform(
        input_df[["Geography"]]
    )

    # Convert sparse to dense if needed
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()

    geo_columns = one_hot_encoder_geography.get_feature_names_out()

    geo_df = pd.DataFrame(geo_encoded, columns=geo_columns)

    # Merge
    input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

    # Drop original Geography column
    input_df.drop(columns=["Geography"], inplace=True)

    # ----------------------------
    # Match EXACT training order
    # ----------------------------
    input_df = input_df[scaler.feature_names_in_]

    # ----------------------------
    # Scale
    # ----------------------------
    input_scaled = scaler.transform(input_df)

    # ----------------------------
    # Predict (Keras uses predict)
    # ----------------------------
    probability = model.predict(input_scaled)[0][0]
    prediction = 1 if probability > 0.5 else 0

    # ----------------------------
    # Display Result
    # ----------------------------
    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.write(f"Churn Probability: {probability:.4f}")