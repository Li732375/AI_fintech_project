import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader.data as WebData

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


# 下載聯邦基金利率數據
fed_funds_rate = WebData.DataReader('FEDFUNDS', 'fred', start = '2020-01-01', 
                                    end = '2024-07-30')
# 顯示數據
excel_filename = 'Fed_Funds_Rate.xlsx'
fed_funds_rate.to_excel(excel_filename)
print(f"匯率資料已存儲為 '{excel_filename}'")
print(fed_funds_rate)

fed_funds_rate['FEDFUNDS'].plot() # 用 FEDFUNDS 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("FEDFUNDS") # y 軸的標籤
plt.title("Fed_Funds_Rate") # 圖標題
plt.show()

