# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:48:24 2024

@author: user
"""

import yfinance as yf
import matplotlib.pyplot as plt
Currency_symbol = 'TWD%3DX' # 輸入股票代號下載匯率資料
Currency_data = yf.download(Currency_symbol, 
                            start = '2020-01-01', end = '2024-07-30') # 獲取特定日期範圍的匯率資料
excel_filename = f'{Currency_symbol}_Currency_data.xlsx' # 將匯率資料存儲為 Excel 檔案，以匯率代號作為檔案名稱
Currency_data.to_excel(excel_filename)
print(f"匯率資料已存儲為 '{excel_filename}'")
print(Currency_data)

Currency_data['Close'].plot() # 用 CLOSE 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("Closing Price") # y 軸的標籤
plt.title(f"{Currency_symbol} Currency Price") # 圖標題
plt.show()