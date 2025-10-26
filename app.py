import streamlit as st
import pandas as pd
import pickle

# Load your pickled scaler and trained model
with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('random_forest.pkl', 'rb') as f:
    model_rf = pickle.load(f)
# PCA is NOT used at prediction time because model was not trained with PCA

st.title("Wine Type Prediction App")

# List of all features used during training (12 features, order matters)
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
    'pH', 'sulphates', 'alcohol', 'quality'
]

st.sidebar.header("Input Wine Features")

input_data = {}
for feature in feature_names:
    if feature == 'quality':
        input_data[feature] = st.sidebar.number_input(
            f"Input {feature}", min_value=0, max_value=10, value=5, step=1)
    else:
        input_data[feature] = st.sidebar.number_input(
            f"Input {feature}", value=0.0)

input_df = pd.DataFrame([input_data])
st.subheader("Your Input Data")
st.write(input_df)

if st.button("Predict Wine Type"):
    scaled_data = scaler.transform(input_df)  # Only scale, NO PCA
    prediction = model_rf.predict(scaled_data)[0]
    wine_type = "Red Wine" if prediction == 0 else "White Wine"
    st.success(f"Predicted Wine Type: {wine_type}")
