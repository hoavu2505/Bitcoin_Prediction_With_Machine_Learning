
from __future__ import division, print_function, unicode_literals
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

## Load training data 
df = pd.read_csv('Training set.csv')

# Unix-time to 
#df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

# Lấy cột Timestamp
X1 = np.array(df[['Timestamp']])

# Lấy cột Volume
X2 = np.array(df[['Volume']])

# Lấy cột Open
X3 = np.array(df[['Open']])

# Lấy cột Market Cap
X4 = np.array(df[['Market Cap']])

# Lấy cột Close (Giá của một ngày)
y = np.array(df.drop(['Date','Timestamp','Open','High','Low','Volume','Market Cap'], axis = 1))

# Build Xbar
one = np.ones((X2.shape[0], 1))
Xbar = np.concatenate((one, X1, X2, X3, X4), axis = 1)

from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

w = regr.coef_
w = w.T

#print( 'Solution found by scikit-learn  : ', w )

w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]

# ## Load test data
# df1 = pd.read_csv('Test.csv')

# # Lấy cột Timestamp
# Test_1 = np.array(df1[['Timestamp']])

# # Lấy cột Volume
# Test_2 = np.array(df1[['Volume']])

# # Lấy cột Open
# Test_3 = np.array(df1[['Open']])

# # Lấy cột Market Cap
# Test_4 = np.array(df1[['Market Cap']])


# z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4

# # print('Prediction: ', z)

# df1['Prediction'] = z

# price = np.array(df1.drop(['Date','Timestamp','Open','High','Low','Volume','Market Cap'], axis = 1))

# print(price)





#GUI
import streamlit as st
import yfinance as yf
from datetime import datetime, date

st.write("""
# Dự đoán giá Bitcoin
""")

st.write("""
## Giá mở
""")
st.line_chart(df.Open)

st.write("""
## Giá đóng
""")
st.line_chart(df.Close)

st.write("""
## Số lượng giao dịch trong ngày
""")
st.area_chart(df.Volume)

file = st.file_uploader("Pick a file")

if file:
    df1 = pd.read_csv(file)
    # Lấy cột Timestamp
    Test_1 = np.array(df1[['Timestamp']])

    # Lấy cột Volume
    Test_2 = np.array(df1[['Volume']])

    # Lấy cột Open
    Test_3 = np.array(df1[['Open']])

    # Lấy cột Market Cap
    Test_4 = np.array(df1[['Market Cap']])
    if st.button('Dự đoán'):
        z = w_0 + w_1*Test_1 + w_2*Test_2 + w_3*Test_3 + w_4*Test_4
        df1['Prediction'] = z       # Thêm một cột Prediction
        st.dataframe(df1)   # output ra bảng
        # Xử lý dữ liệu in ra biểu đồ
        price = pd.DataFrame(df1.drop(['Timestamp','Open','High','Low','Volume','Market Cap'], axis = 1))
        price = price.rename(columns={'Date':'index'}).set_index('index')
        st.write("""
        ## So sánh giá Close và giá Prediction
        """)
        st.line_chart(price)

# # Pick date
# date_string = st.date_input("Chọn ngày dự đoán")    # datetime.date

# # convert datetime.date to datetime.datetime
# date = datetime.strptime(str(date_string), "%Y-%m-%d")

# # convert datetime.datetime to timestamp
# timestamp = datetime.timestamp(date)



