import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

# 加載數據集
data = pd.read_excel('外匯data.xlsx')  # 替換為你的數據文件
X = data.drop(['LABEL', 'Date'], axis=1)
y = data['LABEL']

# 數據縮放
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 轉換數據格式為3D (樣本數，時間步，特徵數)
time_steps = 10  # 假設每個樣本有10個時間步，你可以根據實際情況調整
X_reshaped = []
y_reshaped = []

for i in range(len(X_scaled) - time_steps):
    X_reshaped.append(X_scaled[i:i+time_steps])
    y_reshaped.append(y_scaled[i+time_steps])

X_reshaped = np.array(X_reshaped)
y_reshaped = np.array(y_reshaped)

# 假設我們將數據分為4個批次
batch_size = len(X_reshaped) // 4

# 初始化模型
model = Sequential()
model.add(LSTM(50, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 分批訓練模型
for i in range(4):
    start = i * batch_size
    end = (i + 1) * batch_size
    X_batch = X_reshaped[start:end]
    y_batch = y_reshaped[start:end]
    
    if i == 0:
        # 首次訓練
        model.fit(X_batch, y_batch, epochs=20, batch_size=32, validation_split=0.2)
        #if model.score < model-1.score: #捨棄小於0的資料
        
           #else:繼續訓練



    else:
        # 繼續訓練
        model.fit(X_batch, y_batch, epochs=20, batch_size=32, validation_split=0.2)
    
    # 保存中間模型
    model.save(f'lstm_model_batch_{i+1}.h5')

# 最終保存完整模型
model.save('lstm_model_final.h5')

# 從文件中加載最終模型
#資料分批保存模型final_model = load_model('lstm_model_final.h5')

# 使用加載的模型進行預測
# 這裡使用全部測試數據進行預測
X_test = X_reshaped  # 假設全部數據用作測試
predictions = final_model.predict(X_test)
print(predictions)