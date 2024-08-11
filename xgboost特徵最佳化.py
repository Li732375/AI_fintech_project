import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.dates as mdates


# 讀取數據
df = pd.read_excel('data.xlsx')
df['Index'] = range(len(df))

print(df.columns)
feature_names = ['Close_y', 'High_y', 'CPIAUCNS', 'Open_y', 'UNRATE', 'MA_20', 
                 'MA_10', 'Growth Rate_x', 'TW_CPI_Rate', 
                 'WILLR', 'Open_x', 'K', 'RSI_14', 'Volume_y', 
                 'Growth Rate_y', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate', 'Index']
# 0.821
    
def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
    X = stock_data[feature_names].values
    y = stock_data[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state)

    return X_train, X_test, y_train, y_test, feature_names

label_column = 'LABEL'
delete_column = ['LABEL', 'Volume_x', 'Next_5Day_Return']
accuracies = []

# 分割資料
trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, 
                                                delete_column)
Xgboost = XGBClassifier()
start_time = time.time()
Xgboost.fit(trainX, trainY)
training_time = time.time() - start_time

test_predic = Xgboost.predict(testX) # 取得預測的結果
test_acc = Xgboost.score(testX, testY)
print(f"資料型態 {type(testX)}")
print(testX)

# 進行 XOR 運算
xor_result = np.bitwise_xor(test_predic, testY)
# =============================================================================
# test_predic 和 testY 之間的 XOR 運算可以幫助你找出模型預測錯誤的情況：
# 
# 如果 test_predic[i] 和 testY[i] 相同（即模型預測正確），則 XOR 結果為 0。
# 如果 test_predic[i] 和 testY[i] 不同（即模型預測錯誤），則 XOR 結果為 1。
# =============================================================================
print(f"結果資料型態 {type(xor_result)}")
print(f"結果資料筆數 {len(xor_result)}")
#print(f"XOR 運算結果:\n{xor_result}")
    
print('Xgboost測試集準確率 %.3f' % test_acc)
print(f"訓練時間: {training_time // 60:.2f} 分 {training_time % 60:.2f} 秒")
# 0.821


import matplotlib.colors as mcolors

# 有一維的二元數據（長度不定）
result = pd.Series(xor_result)

def darw(result):
    num_columns = 5 # 每列 5 筆數據
    num_rows = (len(result) + num_columns - 1) // num_columns # 計算需要的行數
    data_padded = np.pad(result, 
                         (0, num_rows * num_columns - len(result)), 
                         mode = 'constant', constant_values = -1) # 補齊數據
    
    # 重塑為每列 5 筆數據的二維矩陣
    result_2d = data_padded.reshape(num_rows, num_columns).T  # 轉置以符合每列顯示的要求

# =============================================================================
#     plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
#     plt.xlabel("週") # x 軸的標籤
#     plt.ylabel("日") # x 軸的標籤
# =============================================================================
     
    # 添加自訂圖例
    #legend_labels = ['補齊部分', '0', '1']
    colors = ['black', 'forestgreen', 'silver']
    cmap = mcolors.ListedColormap(colors) # 自訂顏色映射，補齊、0(對) 、1(錯)
    bounds = [-1, 0, 1, 2] # -1：補齊數據, 0：原始數據, 1：補齊數據
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
       
    # 繪製圖像
    plt.imshow(result_2d, cmap = cmap, norm = norm, interpolation='none', 
               aspect='equal')
    
# =============================================================================
#     # 設置 x 軸和 y 軸的刻度位置
#     plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
#     plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
#     
# =============================================================================
    # 取消顯示刻度
    plt.axis('off')
    
    # 設置刻度字體大小
    #plt.xticks(fontsize = 5)
    #plt.yticks(fontsize = 5)
    
    # 顯示網格線
    #plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
# =============================================================================
#     # 創建圖例框
#     legend_elements = [plt.Line2D([0], [0], 
#                                   color = color, 
#                                   lw = 4, 
#                                   label = label) for color, 
#                        label in zip(colors, legend_labels)]
#     # =============================================================================
# =============================================================================
    # plt.Line2D：
    # 
    # plt.Line2D([0], [0], color=color, lw=4, label=label) 創建了一個線條對象，用於圖例。
    # color=color 設定線條顏色。
    # lw=4 設定線條寬度。
    # label=label 設定圖例標籤。
    # =============================================================================
    
    # 添加圖例
    # =============================================================================
    # plt.legend(handles = legend_elements, loc = 'upper left', 
    #            bbox_to_anchor = (-0.1, 4)) 
    # =============================================================================
    
    #plt.title('XOR 運算結果')
    
darw(result[:])


print(testX.shape)
print(len(feature_names))
# 將 numpy.ndarray 接回成 DataFrame
train_df = pd.DataFrame(trainX, columns = feature_names)
#train_df['target'] = trainY.values  # 將目標變數接回

test_df = pd.DataFrame(testX, columns = feature_names)
#test_df['target'] = testY.values  # 將目標變數接回

# 顯示結果
print("Train DataFrame:\n", train_df.head())
print("\nTest DataFrame:\n", test_df.head())

# 確保 Date 列已經轉換為 DateTime 類型，並設置為索引
df.index = pd.to_datetime(df.index)
df = df.resample('W').agg({'Open_x': 'first', 'High_x': 'max', 'Low_x': 'min', 'Close': 'last'})

# 創建一個新的 DataFrame 用於 K 線圖
df['Color'] = df.apply(lambda row: 'g' if row['Close'] > row['Open_x'] else 'r', axis=1)

# 繪製 K 線圖
plt.figure(figsize=(12, 6))

# 繪製 K 棒
for i in range(len(df)):
    row = df.iloc[i]
    color = row['Color']
    plt.plot([df.index[i], df.index[i]], [row['Low_x'], row['High_x']], color=color, linewidth=1)  # 垂直線
    plt.plot([df.index[i], df.index[i]], [row['Open_x'], row['Close']], color=color, linewidth=5)  # K 棒

# 設置 x 軸格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

plt.xlabel('日期')
plt.ylabel('價格')
plt.title('周K線圖')
plt.xticks(rotation=45)
plt.grid(True)

plt.show()