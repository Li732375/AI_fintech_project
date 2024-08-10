import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.colors import LinearSegmentedColormap

# 讀取數據
df = pd.read_excel('data.xlsx', index_col='Date')
print(df.columns)

def split_stock_data(stock_data, label_column, delete_column, test_size=0.3, random_state=42):
    X = stock_data.drop(delete_column, axis=1)
    feature_names = X.columns.tolist()
    y = stock_data[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, feature_names

label_column = 'LABEL'
delete_column = ['LABEL', 'Volume_x', 'Next_5Day_Return']
trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, delete_column)
model_accuracies = {}

# 訓練 XGBoost 模型
Xgboost = XGBClassifier()
start_time = time.time()
Xgboost.fit(trainX, trainY)
training_time = time.time() - start_time

test_predic = Xgboost.predict(testX)
test_acc = Xgboost.score(testX, testY)
model_accuracies['XGBoost'] = test_acc
print('Xgboost測試集準確率 %.2f' % test_acc)
print(f"測試時間: {training_time:.2f} 秒")

# 創建年分
df['Year'] = df.index.year

# 根據時間範圍和年份分割數據
years = sorted(df['Year'].unique())

# 確保有四年的數據
if len(years) != 4:
    raise ValueError("數據必須包含四年的數據")

# 創建每年的熱量圖
for i, year in enumerate(years):
    plt.figure(figsize=(12, 4))
    
    # 篩選該年份的數據
    year_data = df[df['Year'] == year]
    year_indices = year_data.index
    year_testY = testY[testX.index.isin(year_indices)]
    year_test_predic = test_predic[testX.index.isin(year_indices)]

    # 將實際值和預測值組合成熱量圖數據
    result = [0 if x == y else 1 for x, y in zip(year_testY, year_test_predic)]
    year_heatmap_data = np.array(result).reshape(1, -1)  # 轉換為 (1, N) 的形狀

    # 繪製每年的熱量圖，使用黑白顏色映射
    plt.imshow(year_heatmap_data, cmap='binary', interpolation='nearest', aspect='auto', extent=[0, len(year_testY), 0, 1])
    
    # 添加顏色條
    plt.colorbar()
    
    # 設置標題和標籤
    plt.title(f'年份 {year}')
    plt.xlabel('樣本編號')
    plt.ylabel('數據類型')
    plt.yticks([0.5], ['實際 vs 預測'])
    
    # 顯示圖像
    plt.show()
    