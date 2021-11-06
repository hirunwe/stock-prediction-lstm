import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas_datareader as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


import streamlit as st

start = '2010-01-01'
end = '2020-12-31'

st.title("Stock Prediction App")

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = data.DataReader(user_input, 'yahoo', start, end)

#Describing data

st.subheader('Data from 2010 to 2020')
st.write(df.describe())

#visualizations

st.subheader('Closing price vs. Time graph')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Year')
plt.ylabel('Closing Price ($)')
st.pyplot(fig)


st.subheader('Closing price vs. Time graph with 100 Moving Average')
fig = plt.figure(figsize=(12,6))
ma100 = df.Close.rolling(100).mean()
plt.plot(df.Close, label= 'Actual Closing Price')
plt.plot(ma100, 'r', label= 'Moving Average (100 days)')
plt.xlabel('Year')
plt.ylabel('Closing Price ($)')
plt.legend()
st.pyplot(fig)


st.subheader('Closing price vs. Time graph with 100 and 200 Moving Averages')
fig = plt.figure(figsize=(12,6))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(df.Close,label= 'Actual Closing Price')
plt.plot(ma100, 'r',label= 'Moving Average (100 days)')
plt.plot(ma200, 'g',label= 'Moving Average (200 days)')
plt.xlabel('Year')
plt.ylabel('Closing Price ($)')
plt.legend()
st.pyplot(fig)

#splitting data intro training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#loading the model

model = load_model("ml_modal.h5")

#testing the model
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Visualize the predictions
st.subheader('Predictions vs. the Actual Value')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Actual Value')
plt.plot(y_predicted, 'r', label = 'Predicted Value')
plt.xlabel('Time')
plt.ylabel('Value $')
plt.legend()
st.pyplot(fig2)

