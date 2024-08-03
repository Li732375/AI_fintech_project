# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:46:55 2024

@author: 郭昱
"""

import pandas as pd

Currency_data = pd.read_excel('TWD%3DX_Currency_data.xlsx', 
                              index_col = 'Date')  # 讀取匯率資料

# 處理 y 資料
Currency_data['Next_5Day_Return'] = \
    Currency_data['Open'].diff(5).shift(-5) # 計算價格變化

def classify_return(x):
    return 1 if x > 0 else 0  # 標示漲跌，大於0標示為漲(1)，小於0標示為跌(0)

Currency_data['LABEL'] = \
    Currency_data['Next_5Day_Return'].apply(
        classify_return) # 創造新的一列 LABEL 來記錄漲跌
Currency_data = Currency_data.dropna() # 刪除因技術指標計算出現的空值
Currency_data.to_excel("試跑資料TWD%3DX_Currency_data.xlsx") # 將整理好的資料存成excel



ones_count = (Currency_data['LABEL'] == 1).sum()
zero_count = (Currency_data['LABEL'] == 0).sum()
print(f"上漲數為 {ones_count}")
print(f"下跌數為 {zero_count}")