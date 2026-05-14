import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Load trained models and columns
with open('bank_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('bank_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('bank_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("🏦 Bank Deposit Prediction App")
st.write("Enter the customer details below to check if they will subscribe to a term deposit:")

# 2. Creating Input Fields based on bank.csv data structure
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    balance = st.number_input("Annual Balance (in EUR)", min_value=-5000, max_value=100000, value=1000)
    day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)
    duration = st.number_input("Contact Duration (seconds)", min_value=0, max_value=5000, value=200)
    campaign = st.number_input("Number of Contacts during Campaign", min_value=1, max_value=50, value=1)
    pdays = st.number_input("Days since last contact (-1 means never)", min_value=-1, max_value=1000, value=-1)
    previous = st.number_input("Number of contacts before campaign", min_value=0, max_value=50, value=0)

with col2:
    job = st.selectbox("Job Type", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'self-employed', 'student', 'unknown'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.selectbox("Education Level", ['secondary', 'tertiary', 'primary', 'unknown'])
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    contact = st.selectbox("Contact Communication Type", ['cellular', 'unknown', 'telephone'])
    month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    poutcome = st.selectbox("Outcome of Previous Marketing Campaign", ['unknown', 'other', 'failure', 'success'])

# 3. Process Input Data when Predict button is clicked
if st.button("Predict Deposit Subscription"):
    # Create raw DataFrame from inputs
    input_dict = {
        'age': age, 'job': job, 'marital': marital, 'education': education, 
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan, 
        'contact': contact, 'day': day, 'month': month, 'duration': duration, 
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }
    input_df = pd.DataFrame([input_dict])

    # Convert text data using One-Hot Encoding just like we did in training
    input_encoded = pd.get_dummies(input_df)
    
    # Ensure all columns match the original model columns layout
    final_features = pd.DataFrame(columns=model_columns)
    final_features = pd.concat([final_features, input_encoded], axis=0).fillna(0)
    final_features = final_features[model_columns] # Reorder columns exactly

    # Scale the inputs
    final_features_scaled = scaler.transform(final_features)

    # Prediction
    prediction = model.predict(final_features_scaled)

    # Display Result
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🎉 **Yes!** The customer is highly likely to subscribe to the term deposit.")
    else:
        st.error("❌ **No.** The customer is unlikely to subscribe to the term deposit.")
