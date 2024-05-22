import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Load the trained model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Define the prediction function
def predict_performance(design_flow, actual_flow):
    # Create a DataFrame with the input values
    input_df = pd.DataFrame({'Design_flow(l/s)': [design_flow], 'Actual_flow(l/s)': [actual_flow]})
    # Use the model to make a prediction
    prediction = loaded_model.predict(input_df)
    # Round the prediction
    rounded_prediction = np.round(prediction)
    return rounded_prediction[0]

# Streamlit app
def main():
    st.title("FCU Performance Prediction")
    
    st.write("""
    ## Enter the design and actual flow values to predict the performance percentage:
    """)
    
    design_flow = st.number_input("Design Flow (l/s)", min_value=0.0, step=0.1, format="%.2f")
    actual_flow = st.number_input("Actual Flow (l/s)", min_value=0.0, step=0.1, format="%.2f")
    
    if st.button("Predict"):
        result = predict_performance(design_flow, actual_flow)
        st.write(f"Predicted Performance (%): {result}")

if __name__ == "__main__":
    main()
