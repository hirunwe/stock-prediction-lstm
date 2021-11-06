import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px
import pandas_datareader as data
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import streamlit as st

start = '2010-01-01'
end = '2020-12-31'

st.title("Stock Prediction App")

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = data.DataReader(user_input, 'yahoo', start, end)
df['Date'] = df.index
df['MA100'] = df.Close.rolling(100).mean()
df['MA200'] = df.Close.rolling(200).mean()

#Describing data

st.subheader('Data from 2010 to 2020')
st.write(df.describe())

#visualizations
#1st Plot
fig = px.line(df,x="Date",y="Close",title = 'Closing price vs. Time graph')
fig.update_layout(xaxis_title='Year',
                    yaxis_title='Closing Price ($)',
                   template="simple_white")
st.plotly_chart(fig)

#2nd Plot

fig2 = go.Figure()
fig2.update_layout(title = 'Closing price vs. Time graph with 100 Moving Average', xaxis_title='Year',
                   yaxis_title='Closing Price ($)',
                   template="simple_white")
fig2.add_trace(go.Scatter(x=df.Date, y=df.Close, 
                          line=dict(color='blue', width=1),
                          name='Actual Closing Price'))
fig2.add_trace(go.Scatter(x=df.Date, y=df.MA100, 
                          line=dict(color='orange', width=1),
                          name='Moving Average (100 days)'))
fig2.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
st.plotly_chart(fig2)

#3rd PLot

fig3 = go.Figure()
fig3.update_layout(title = 'Closing price vs. Time graph with 100 and 200 Moving Averages', xaxis_title='Year',
                   yaxis_title='Closing Price ($)',
                   template="simple_white")
fig3.add_trace(go.Scatter(x=df.Date, y=df.Close, 
                          line=dict(color='blue', width=1),
                          name='Actual Closing Price'))
fig3.add_trace(go.Scatter(x=df.Date, y=df.MA100, 
                          line=dict(color='red', width=1),
                          name='Moving Average (100 days)'))
fig3.add_trace(go.Scatter(x=df.Date, y=df.MA200, 
                          line=dict(color='green', width=1),
                          name='Moving Average (200 days)'))
fig3.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
st.plotly_chart(fig3)


#splitting data intro training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#loading the model

model = load_model(r'C:\Users\hirun\Documents\Stock-Final-Project\ml_modal.h5')

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
y_predicted = y_predicted.flatten() 

#Visualize the predictions

fig5 = go.Figure()
fig5.update_layout(title = 'Predictions vs. the Actual Value', xaxis_title='Time',
                   yaxis_title='Value $',
                   template="simple_white")
fig5.add_trace(go.Scatter(y=y_test, 
                          line=dict(color='blue', width=1),
                          name='Actual Value'))
fig5.add_trace(go.Scatter(y=y_predicted, 
                          line=dict(color='orange', width=1),
                          name='Predicted Value'))
fig5.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
st.plotly_chart(fig5)

