import streamlit as st
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
import joblib
import numpy as np

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Function to predict donation likelihood
def predict_donation(features):
    # Preprocess the features (normalize them as needed)
    # Here you would have to do the same preprocessing you did for training
    # For demonstration purposes, I'll just return the input for now
    return features

# Streamlit app
def main():
    st.title('Blood Donation Prediction')

    # Input features
    st.header('Enter Donor Information:')
    recency = st.slider('Recency (months)', 0, 50, 5)
    frequency = st.slider('Frequency (times)', 0, 50, 5)
    time = st.slider('Time (months)', 0, 100, 10)
    monetary = st.slider('monetary_log', 0, 100, 5)


    # Predict button
    if st.button('Predict'):
        # Create a dataframe for prediction
        input_data = pd.DataFrame({
            'Recency (months)': [recency],
            'Frequency (times)': [frequency],
            'Time (months)': [time],
            'monetary_log': [np.log(monetary)],            
        })

        # Make prediction
        prediction = model.predict_proba(input_data)[:, 1]

        # Display prediction
        st.subheader('Prediction Probability')
        st.write(f"The likelihood of donating blood is: {prediction[0]:.2f}")

        # Additional details
        st.subheader('Additional Details:')
        if prediction[0] >= 0.5:
            st.write("This donor is more likely to donate blood.")
        else:
            st.write("This donor is less likely to donate blood.")

# Run the app
if __name__ == '__main__':
    main()
