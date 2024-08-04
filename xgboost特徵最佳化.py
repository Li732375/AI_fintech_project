import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_excel('外匯data.xlsx',index_col = 'Date')

def split_stock_data(stock_data, label_column, delete_column, test_size = 0.3, 
                     random_state = 42):
# =============================================================================
#     feature_names = ['Bollinger Bands lower_5', 'MA_5_15', 'MA_20_10', 'Open', 
#                      'Bollinger Bands lower_20', 'MA_20_5', 'MA_10', 'MA_20_15', 
#                      'MA_5', 'MACD_5', 'MA_20_20', 'MA_5_5', 'KD_15', 
#                      'Bollinger Bands Upper_15', 'Bollinger Bands lower', 
#                      'MACD', 'MA_10_20', 'MACD_15', 'RSI_14', 'KD', 
#                      'Bollinger Bands lower_15', 'MA_20', 
#                      'Bollinger Bands Upper_10', 'MACD_20']
# =============================================================================
    feature_names = ['Bollinger Bands lower_5', 'MA_20_20', 'MA_5_5', 
                     'Bollinger Bands lower', 'MA_10_20', 'RSI_14', 'MACD_20']
 
    X = stock_data[feature_names].values
    y = stock_data[label_column].values # y為標籤(LABEL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state) # 資料分割

    return X_train, X_test, y_train, y_test, feature_names

label_column = 'LABEL' # 標籤欄位
delete_column = 'Next_5Day_Return' # 刪除的欄位
trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, 
                                                delete_column)

model_accuracies = {}

import time


Xgboost = XGBClassifier()
start_time = time.time()
Xgboost.fit(trainX, trainY)
training_time = time.time() - start_time

test_acc = Xgboost.score(testX, testY)
model_accuracies['XGBoost'] = test_acc
print('Xgboost測試集準確率 %.4f' % test_acc)
print(f"測試時間: {training_time:.4f} 秒")


# 繪製特徵重要性圖
import matplotlib.pyplot as plt

# 將特徵名稱和重要性配對
feature_importance_pairs = list(zip(feature_names, 
                                    Xgboost.feature_importances_))

sorted_pairs = sorted(feature_importance_pairs, key = lambda x: x[1], 
                      reverse = True)

# 提取排序後的特徵，[:] 取得前幾名的特徵和重要性
sorted_feature_names, sorted_importances = zip(*sorted_pairs[:])

# 繪製特徵重要性橫條圖
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 設置中文字體
plt.figure(figsize = (12, 8))
bars = plt.barh(sorted_feature_names, sorted_importances, color = 'skyblue')
        
# 顯示每個橫條的數值
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{width * 100:.2f} %', 
             va = 'center', ha = 'left', fontsize = 10)
    
plt.xlabel('特徵重要性')
plt.ylabel('特徵')
plt.title('特徵重要性')
plt.tight_layout(pad = 0.5)
plt.gca().invert_yaxis()  # 反轉 y 軸，使重要性高的特徵顯示在上面
plt.show()


best_model = max(model_accuracies, key = model_accuracies.get)
best_accuracy = model_accuracies[best_model]
print(f'準確率最高的模型是 {best_model}，準確率為 %.4f' % best_accuracy)
