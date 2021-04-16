
from __future__ import division, print_function, unicode_literals
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

## Load training data 
df = pd.read_csv('Training set.csv')

# Lấy cột Volume
X1 = np.array(df[['Volume']])

# Lấy cột Open
X2 = np.array(df[['Open']])

# Lấy cột High
X3 = np.array(df[['High']])

# Lấy cột Low
X4 = np.array(df[['Low']])

# Lấy cột Market Cap
X5 = np.array(df[['MarketCap']])

# Lấy cột Close (Giá của một ngày)
y = np.array(df.drop(['Date','Open','High','Low','Volume','MarketCap'], axis = 1))

# Build Xbar
one = np.ones((X2.shape[0], 1))
Xbar = np.concatenate((one, X1, X2, X3, X4, X5), axis = 1)


from sklearn import datasets, linear_model

from sklearn.metrics import r2_score

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

w = regr.coef_
w = w.T

# print( 'Solution found by scikit-learn  : ', w )
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]

import plotly.graph_objects as go
import plotly.express as px

# # Load test data
# df1 = pd.read_csv('Test.csv')

# # Lấy cột Volume
# Test_1 = np.array(df1[['Volume']])

# # Lấy cột Open
# Test_2 = np.array(df1[['Open']])

# # Lấy cột High
# Test_3 = np.array(df1[['High']])

# # Lấy cột Low
# Test_4 = np.array(df1[['Low']])

# # Lấy cột Market Cap
# Test_5 = np.array(df1[['Market Cap']])

# z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4 + w_5*Test_5

# # print('Prediction: ', z)

# df1['Prediction'] = z

# price = np.array(df1.drop(['Date','Open','High','Low','Volume','Market Cap'], axis = 1))

# print(price)





#GUI
import streamlit as st
import yfinance as yf
from datetime import datetime, date


st.write("""
# Dự đoán giá Bitcoin
""")

close_train = px.line(df, x='Date', y='Close', title='Giá Close (USD)')
close_train.update_xaxes(rangeslider_visible=True)
st.plotly_chart(close_train)


volume_train = px.line(df, x='Date', y='Volume', title='Số lượng giao dịch trong ngày (USD)')
volume_train.update_xaxes(rangeslider_visible=True)
st.plotly_chart(volume_train)


market_train = px.line(df, x='Date', y='MarketCap', title='Vốn hóa thị trường (USD)')
market_train.update_xaxes(rangeslider_visible=True)
st.plotly_chart(market_train)

st.write("""
## Biểu đồ trading
""")
trading_train = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
st.plotly_chart(trading_train)

file = st.file_uploader("Chọn tập cần dự đoán")

if file:
    df1 = pd.read_csv(file)
    # Lấy cột Volume
    Test_1 = np.array(df1[['Volume']])

    # Lấy cột Open
    Test_2 = np.array(df1[['Open']])

    # Lấy cột High
    Test_3 = np.array(df1[['High']])

    # Lấy cột Low
    Test_4 = np.array(df1[['Low']])

    # Lấy cột Market Cap
    Test_5 = np.array(df1[['MarketCap']])

    Test_6 = np.array(df1[['Close']])   #Lấy cột Close để tính Accuracy

    if st.button('Dự đoán'):
        st.success('Dự đoán xong. Mời kiểm tra kết quả!')

        z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4 + w_5*Test_5
        
        st.write("Độ chính xác: ",r2_score(Test_6,z))   # Accuracy của LR

        df1['Prediction'] = z       # Thêm một cột Prediction
        st.dataframe(df1)   # output ra bảng
        # Xử lý dữ liệu in ra biểu đồ
        price = pd.DataFrame(df1.drop(['Open','High','Low','Volume','MarketCap'], axis = 1))
        price = price.rename(columns={'Date':'index'}).set_index('index')

        st.write("""
        ## Biểu đồ Trading
        """)
        trading_test = go.Figure(data=[go.Candlestick(x=df1['Date'],
                open=df1['Open'],
                high=df1['High'],
                low=df1['Low'],
                close=df1['Close'])])
        st.plotly_chart(trading_test)

        close_test = px.line(df1, x='Date', y='Close', title='Biểu đồ giá Close')
        close_test.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(close_test)

        #prediction_test = go.Figure([go.Scatter(x=df1['Date'], y=df1['Prediction'], line_color="#ff8303" )])
        prediction_test = px.line(df1, x='Date', y='Prediction', title='Biểu đồ giá Prediction')
        prediction_test.update_traces(line_color='#ff8303') #Đổi màu đường line
        prediction_test.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(prediction_test)

        st.write("""
        ## So sánh giá Close và giá Prediction
        """)
        st.line_chart(price)

        







