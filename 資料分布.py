# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:30:33 2024

@author: 郭昱
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_excel("試跑資料TWD%3DX_Currency_data.xlsx")

# 計算每個標籤的比例
label_proportions = df['LABEL'].value_counts(normalize = True)
print("全部資料分布：")
print(label_proportions)

print('--------------')
# 假設資料中有一個標籤欄位（例如 '標籤'），用於保持資料分布
labels = df['LABEL']
# 設定分割數 N
N = 4  # 例如，將資料分割成 4 筆

# 初始化 StratifiedKFold
skf = StratifiedKFold(n_splits = N, shuffle = True, 
                      random_state = 42)

# 創建一個列表來儲存分割後的 N 筆資料
split_data = [pd.DataFrame() for _ in range(4)]

# 分割資料並保持分布
for i, (_, test_index) in enumerate(skf.split(df, labels)):
    split_data[i] = df.iloc[test_index]

# 檢查分割後的資料分布
for i, split in enumerate(split_data):
    print(f"分割資料 {i+1}：")
    print(split['LABEL'].value_counts(normalize = True))
    split_data.to_excel(f"外匯分割資料_{i+1}.xlsx", index = False)

    
