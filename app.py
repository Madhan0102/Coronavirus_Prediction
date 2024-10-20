import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Load the machine learning model
model = pickle.load(open('xyz.pkl', 'rb'))

# Title of the app
st.title('COVID-19 Data Predictions')

# Collect user inputs
sno = st.number_input('Serial Number (Sno)', min_value=0, step=1)
date = st.date_input('Date')
time = st.time_input('Time')
state = st.text_input('State/Union Territory')
confirmed_indian_national = st.number_input('Confirmed Indian National', min_value=0, step=1)
confirmed_foreign_national = st.number_input('Confirmed Foreign National', min_value=0, step=1)
cured = st.number_input('Cured', min_value=0, step=1)
deaths = st.number_input('Deaths', min_value=0, step=1)
confirmed = st.number_input('Confirmed', min_value=0, step=1)

# Convert date and time to suitable numerical format
date_num = (datetime.combine(date, datetime.min.time()) - datetime(1970, 1, 1)).days  # Days since epoch
time_num = time.hour * 60 + time.minute  # Total minutes since midnight

# Create a dataframe from the inputs
input_data = pd.DataFrame({
    'Sno': [sno],
    'Date': [date_num],  # Use the numerical format of date
    'Time': [time_num],  # Use the numerical format of time
    'State/UnionTerritory': [state],
    'ConfirmedIndianNational': [confirmed_indian_national],
    'ConfirmedForeignNational': [confirmed_foreign_national],
    'Cured': [cured],
    'Deaths': [deaths],
    'Confirmed': [confirmed]
})

# Button to predict
if st.button('Predict'):
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f'Prediction: {prediction}')

    # Plotting the prediction
    st.bar_chart(pd.DataFrame(prediction, columns=['Prediction']))
