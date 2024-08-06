import yfinance as yf
import matplotlib.pyplot as plt

Data_Time_Start = '2020-01-01'
Data_Time_End = '2023-12-31'
Data_Time_TW_Start = str(int(Data_Time_Start[0 : 4]) - 1911) + '-01-01'
Data_Time_TW_End = str(int(Data_Time_End[0 : 4]) - 1911) + '-12-31'

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
plt.title("USD -> TWD") # 圖標題
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

excel_filename = 'USA_CPI_Data.xlsx'
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

excel_filename = 'USA_Unemployment_Rate.xlsx'
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

# 消費者物價指數及其年增率 網址
url = 'https://ws.dgbas.gov.tw/001/Upload/463/relfile/10315/2414/cpispl.xls'

# pip install xlrd
# 直接從 URL 讀取 excel 文件
TW_cpi = pd.read_excel(url, header = 2) # 指定第三行（索引為2）作為欄位名稱
print(TW_cpi.columns) # 檢視所有欄位

TW_cpi = TW_cpi.drop(columns = ['累計平均']) # 移除該欄位
TW_cpi = TW_cpi[:-4] # 移除最後四筆資料
print(TW_cpi)

# 轉換為長格式。將指定列變成行，並且通常是將多個列的數據合併成少數幾列
TW_cpi = TW_cpi.melt(id_vars = '民國年', var_name = '月份', 
                     value_name = 'CPI')

# regex 參數的預設值是 True，會將要替換的字串視為正則表達式處理。
TW_cpi['月份'] = TW_cpi['月份'].str.replace('月', '', regex = False) # 轉換月份

# print(TW_cpi[TW_cpi.isna().any(axis = 1)]) # 顯示缺失值資料
TW_cpi['西元年'] = TW_cpi['民國年'] + 1911
TW_cpi = TW_cpi.drop(columns = ['民國年']) # 移除該欄位
TW_cpi['Date'] = TW_cpi['西元年'].astype(str) + '/' + TW_cpi['月份'] + '/1  12:00:00 AM' # 合併兩時間為新欄位
TW_cpi['Date'] = pd.to_datetime(TW_cpi['Date']) # 將 'date_str' 欄位轉換為時間格式
TW_cpi = TW_cpi.drop(columns = ['西元年', '月份']) # 移除該欄位
TW_cpi = TW_cpi.set_index(['Date']) # 設定索引
TW_cpi = TW_cpi.sort_index()
TW_cpi = TW_cpi.loc[Data_Time_Start : Data_Time_End]
print(TW_cpi)


excel_filename = 'TW_CPI.xlsx'
TW_cpi.to_excel(excel_filename)
print(f"台灣 消費者物價指數 資料已存儲為 '{excel_filename}'")

# 顯示數據
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
TW_cpi['CPI'].plot() # 畫出圖形
plt.xlabel("Date") # x 軸的標籤
plt.ylabel("CPI") # y 軸的標籤
plt.title("台灣 消費者物價-指數") # 圖標題
plt.show()