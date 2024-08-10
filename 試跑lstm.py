import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import time


# 加載數據集
data = pd.read_excel('data.xlsx')
print(data.columns)

# =============================================================================
# feature_names = ['FEDFUNDS', 'BOP', 'WILLR', 'WMA', 'AVGPRICE', 'D', 'LINEARREG_ANGLE', 'MA_10', 'CCI', 'Close', 'SAR', 'MA_20', 'K', 'High', 'MOM', 'WCLPRICE', 'Low', 'STDDEV', 'Bollinger Bands lower', 'MACD', 'RSI_14', 'UNRATE', 'CPIAUCNS', 'Bollinger Bands Upper', 'Open', 'CPI', 'MA_5']
# X = data[feature_names].values
# =============================================================================
X = data.drop(['LABEL', 'Date', 'Volume'], axis = 1)
y = data['LABEL']

# 數據縮放
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 轉換數據格式為3D (樣本數，時間步，特徵數)
time_steps = 10  # 假設每個樣本有10個時間步
X_reshaped = []
y_reshaped = []

for i in range(len(X_scaled) - time_steps):
    X_reshaped.append(X_scaled[i : i + time_steps])
    y_reshaped.append(y_scaled[i + time_steps])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# 分割數據集
X_train, X_test, Y_train, Y_test = train_test_split(X_reshaped, 
                                                    y_reshaped, 
                                                    test_size = 0.3, 
                                                    random_state = 42)

units_1 = 256
adam_learning_rate = 0.001
epochs = 350
batch_size = 16

# 構建LSTM模型
model = Sequential()
model.add(LSTM(units = units_1, return_sequences = True, 
               input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation = 'relu'))
model.compile(optimizer = Adam(learning_rate = adam_learning_rate), 
              loss = 'mean_squared_error', metrics = ['accuracy']) # 編譯模型

start_time = time.time()
# 訓練模型
model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, 
          validation_split = 0.2)
training_time = time.time() - start_time

# 保存模型
model_Save_Name = 'lstm_model.h5'
model.save(model_Save_Name)
print(f"成功保存模型 {model_Save_Name}")

# 加載模型
loaded_model = tf.keras.models.load_model('lstm_model.h5')
print(f"成功載入模型 {model_Save_Name}")

# 使用加載的模型進行預測
predictions = loaded_model.predict(X_test)

train_loss, train_acc = model.evaluate(X_train, Y_train, verbose = 0) # 訓練集準確度計算
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 0) # 測試集準確度計算
print('LSTM訓練集準確率: %.3f' % train_acc)
print('LSTM測試集準確率: %.3f' % test_acc)
print(f"訓練時間: {training_time // 60:.2f} 分 {training_time % 60:.2f} 秒")
print(f"參數 \
      \n'time_steps'： {time_steps} \
      \n'units_1'： {units_1} \
      \n'adam_learning_rate'： {adam_learning_rate} \
      \n'epochs'： {epochs} \
      \n'batch_size'： {batch_size}")

