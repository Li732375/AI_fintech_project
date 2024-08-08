import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

# 加載數據集
data = pd.read_excel('data.xlsx')
X = data.drop(['LABEL', 'Date'], axis = 1)
y = data['LABEL']
print(data.head())
print(data.columns)
print(f"shape {data.shape}")

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

# 轉換為 pandas DataFrame
X_reshaped_df = pd.DataFrame(X_reshaped)
y_reshaped_df = pd.DataFrame(y_reshaped)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X_reshaped_df, 
                                                    y_reshaped_df, 
                                                    test_size = 0.3, 
                                                    random_state = 42)

# 構建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs = 20, batch_size = 32, 
          validation_split = 0.2)

# 保存模型
model_Save_Name = 'lstm_model.h5'
model.save(model_Save_Name)
print(f"成功保存模型 {model_Save_Name}")

# 加載模型
loaded_model = tf.keras.models.load_model('lstm_model.h5')
print(f"成功載入模型 {model_Save_Name}")

# 使用加載的模型進行預測
predictions = loaded_model.predict(X_test)
print(predictions)