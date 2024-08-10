import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt


df = pd.read_excel('data.xlsx', index_col = 'Date')
print(df.columns)

feature_names = ['Close_y', 'High_y', 'CPIAUCNS', 'Open_y', 'UNRATE', 'MA_20', 
                 'MA_10', 'Growth Rate_x', 'USA_CPI_Rate', 'TW_CPI_Rate', 
                 'WILLR', 'Open_x', 'K', 'RSI_14', 'Volume_y', 
                 'Growth Rate_y', 'FEDFUNDS', 'Bollinger Bands lower', 
                 'Bollinger Bands Upper', 'USA_GDP_Rate']
# 0.821
    
def split_stock_data(stock_data, label_column, delete_column, MoveLabeliIndex, 
                     test_size = 0.3, 
                     random_state = 42):
    feature_name = feature_names[:i] + feature_names[i:] 
    feature_name.remove(feature_name[MoveLabeliIndex])
    
    X = stock_data[feature_name].values
    y = stock_data[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, 
                                                        random_state = 
                                                        random_state)

    return X_train, X_test, y_train, y_test, feature_names

label_column = 'LABEL' # 標籤欄位
# 刪除的欄位
delete_column = ['LABEL', 'Volume_x', 'Next_5Day_Return']
accuracies = []


for i in range(len(feature_names) - 1):
    trainX, testX, trainY, testY, feature_names = split_stock_data(df, label_column, 
                                                    delete_column, i)
    Xgboost = XGBClassifier()
    start_time = time.time()
    Xgboost.fit(trainX, trainY)
    training_time = time.time() - start_time
    
    #test_predic = Xgboost.predict(testX) # 取得預測的結果
    test_acc = Xgboost.score(testX, testY)
    accuracies.append(test_acc)
    
    print('Xgboost測試集準確率 %.3f' % test_acc)
    print(f"訓練時間: {training_time // 60:.2f} 分 {training_time % 60:.2f} 秒")

print(f'準確率最高為 {max(accuracies):.3f}, 第 {accuracies.index(max(accuracies)) + 1} 個 , {feature_names[accuracies.index(max(accuracies))]}')
# 0.821