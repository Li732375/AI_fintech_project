# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:04:40 2024

@author: 郭昱
"""

import pandas as pd
import talib

Currency_data = pd.read_excel('TWD%3DX_Currency_data.xlsx', 
                              index_col = 'Date')  # 讀取匯率資料

missing_values = Currency_data.isnull().sum() # 檢查每一列是否有空值

print(missing_values)

Currency_data.drop(columns = ['Adj Close'], inplace = True)
df_close = Currency_data['Close']
df_high = Currency_data['High']
df_low = Currency_data['Low']


# 處理 x 資料
Currency_data['MA_5'] = talib.SMA(df_close, 5) # 計算 MA5
Currency_data['MA_10'] = talib.SMA(df_close, 10) # 計算 MA10
Currency_data['MA_20'] = talib.SMA(df_close, 20) # 計算 MA20
Currency_data['RSI_14'] = talib.RSI(df_close, 14) # 計算 RSI
macd, macdsignal, macdhist = talib.MACD(df_close, fastperiod = 12, 
                                        slowperiod = 26, signalperiod = 9) # 計算 MACD
Currency_data['MACD'] = macd # 將 MACD 計算結果存回資料中
Currency_data['KD'],  Currency_data['KD']= talib.STOCH(df_high, df_low, df_close, fastk_period = 9, 
                                  slowk_period = 3, slowd_period = 3) # 計算 KD
columns_to_shift = ['Close', 'MA_5', 'MA_10', 'MA_20', 'RSI_14', 'MACD', 
                    'KD_K', 'KD_D'] #選取需要進行處理的欄位名稱

upperband, middleband, lowerband = talib.BBANDS(df_close, 
                                          timeperiod=5, 
                                          nbdevup=2, nbdevdn=2, 
                                          matype=0)
Currency_data['Bollinger Bands Upper'] = upperband
Currency_data['Bollinger Bands Middle'] = middleband
Currency_data['Bollinger Bands lower'] = lowerband


columns_to_shift = ['Close', 'MA_5', 'MA_10', 'MA_20', 'RSI_14', 'MACD', 
                    'KD','Bollinger Bands Upper', 'Bollinger Bands Middle' ,
                    'Bollinger Bands lower'] #選取需要進行處理的欄位名稱

# 參考前 5(週), 10(雙週), 15(三週), 20(月) 作為特徵相關參考
for period in range(5, 21,5): # 運用迴圈帶入前 N 期收盤價
        for column in columns_to_shift: # 運用迴圈走訪所選的欄位名稱
            Currency_data[f'{column}_{period}'] = \
                Currency_data[column].shift(period) # 運用.shift()方法取得收盤價

# 處理 y 資料
Currency_data['Next_5Day_Return'] = \
    Currency_data['Close'].diff(5).shift(-5) # 計算價格變化

def classify_return(x):
    return 1 if x > 0 else 0  # 標示漲跌，大於0標示為漲(1)，小於0標示為跌(0)

Currency_data['LABEL'] = \
    Currency_data['Next_5Day_Return'].apply(
        classify_return) # 創造新的一列 LABEL 來記錄漲跌
Currency_data = Currency_data.dropna() # 刪除因技術指標計算出現的空值
Currency_data.to_excel("外匯data.xlsx") # 將整理好的資料存成excel

ones_count = (Currency_data['LABEL'] == 1).sum()
zero_count = (Currency_data['LABEL'] == 0).sum()
print(f"上漲數為 {ones_count}")
print(f"下跌數為 {zero_count}")