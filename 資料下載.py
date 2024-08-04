import yfinance as yf
import matplotlib.pyplot as plt

Data_Time_Start = '2020-01-01'
Data_Time_End = '2023-12-31'

Currency_symbol = 'TWD%3DX' # 輸入股票代號下載匯率資料
Currency_data = yf.download(Currency_symbol, 
                            start = Data_Time_Start, end = Data_Time_End) # 獲取特定日期範圍的匯率資料

excel_filename = f'{Currency_symbol}_Currency_data.xlsx' # 將匯率資料存儲為 Excel 檔案，以匯率代號作為檔案名稱
Currency_data.to_excel(excel_filename)
print(f"匯率資料已存儲為 '{excel_filename}'")
print(Currency_data)

# 顯示數據
Currency_data['Close'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("Closing Price") # y 軸的標籤
plt.title(f"USD -> TWD") # 圖標題
plt.show()


import pandas_datareader.data as WebData

# pip install pandas_datareader
# 下載聯邦基金利率數據
fed_funds_rate = WebData.DataReader('FEDFUNDS', 'fred', 
                                    start = Data_Time_Start, 
                                    end = Data_Time_End)

excel_filename = 'Fed_Funds_Rate.xlsx'
fed_funds_rate.to_excel(excel_filename)
print(f"匯率資料已存儲為 '{excel_filename}'")
print(fed_funds_rate)

# 顯示數據
fed_funds_rate['FEDFUNDS'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("FEDFUNDS") # y 軸的標籤
plt.title("Fed Funds Rate") # 圖標題
plt.show()


# 下載美國 CPI 數據
cpi_data = WebData.get_data_fred('CPIAUCNS',
                                       start = Data_Time_Start, 
                                       end = Data_Time_End)

excel_filename = 'cpi_data.xlsx'
cpi_data.to_excel(excel_filename)
print(f"美國 cpi 資料已存儲為 '{excel_filename}'")
print(cpi_data)

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
cpi_data['CPIAUCNS'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("CPIAUCNS") # y 軸的標籤
plt.title("美國 CPI") # 圖標題
plt.show()


# 下載美國失業率數據
unemployment_rate = WebData.get_data_fred('UNRATE',
                                          start = Data_Time_Start, 
                                          end = Data_Time_End)

excel_filename = 'unemployment_rate.xlsx'
unemployment_rate.to_excel(excel_filename)
print(f"美國失業率資料已存儲為 '{excel_filename}'")
print(unemployment_rate)

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
unemployment_rate['UNRATE'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("UNRATE") # y 軸的標籤
plt.title("美國失業率") # 圖標題
plt.show()


import pandas as pd

# 年度國內主要經濟指標 網址
url = 'https://quality.data.gov.tw/dq_download_csv.php?nid=130489&md5_url=1d30ad9463c9031a71bd9a0ee3cff88d'

# 直接從 URL 讀取 CSV 文件到 DataFrame
TW_cpi = pd.read_csv(url, index_col = '年度')
TW_cpi = TW_cpi.loc[Data_Time_Start[0 : 4] : Data_Time_End[0 : 4]]

excel_filename = 'TW_Indeies.xlsx'
TW_cpi['消費者物價-指數'].to_excel(excel_filename)
print(f"台灣 消費者物價-指數 資料已存儲為 '{excel_filename}'")
print(TW_cpi['消費者物價-指數'].dtypes)
print(TW_cpi.index.dtype)
print(TW_cpi['消費者物價-指數'])

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體

# 將列轉換為數值型，忽略錯誤的值
TW_cpi['消費者物價-指數'] = pd.to_numeric(TW_cpi['消費者物價-指數'], 
                                   errors = 'coerce')
# 將整數列轉換為字串
TW_cpi.index = pd.to_numeric(TW_cpi.index, errors = 'coerce').astype('str')

TW_cpi['消費者物價-指數'].plot() # 畫出圖形
plt.xlabel("年度") # x 軸的標籤
plt.ylabel("消費者物價-指數") # y 軸的標籤
plt.title("台灣 消費者物價-指數") # 圖標題
plt.show()


# =============================================================================
# # 年度國內主要經濟指標 網址
# url = 'https://www.stat.gov.tw/cp.aspx?n=2665'
# 
# # 直接從 URL 讀取 CSV 文件
# TW_cpi = pd.read_csv(url, index_col = '年度')
# TW_cpi = TW_cpi.loc[Data_Time_Start[0 : 4] : Data_Time_End[0 : 4]]
# 
# excel_filename = 'TW_Indeies.xlsx'
# TW_cpi['消費者物價-指數'].to_excel(excel_filename)
# print(f"台灣 消費者物價-指數 資料已存儲為 '{excel_filename}'")
# print(TW_cpi['消費者物價-指數'].dtypes)
# print(TW_cpi.index.dtype)
# print(TW_cpi['消費者物價-指數'])
# 
# # 顯示數據
# plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
# 
# # 將列轉換為數值型，忽略錯誤的值
# TW_cpi['消費者物價-指數'] = pd.to_numeric(TW_cpi['消費者物價-指數'], 
#                                    errors='coerce')
# # 將整數列轉換為整數
# TW_cpi.index = pd.to_numeric(TW_cpi.index, errors = 'coerce').astype('str')
# 
# TW_cpi['消費者物價-指數'].plot() # 畫出圖形
# plt.xlabel("年度") # x 軸的標籤
# plt.ylabel("消費者物價-指數") # y 軸的標籤
# plt.title("台灣 消費者物價-指數") # 圖標題
# plt.show()
# =============================================================================
