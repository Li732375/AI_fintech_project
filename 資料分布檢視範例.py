import pandas as pd
from sklearn.model_selection import StratifiedKFold


df = pd.read_excel("TWD%3DX_Currency_data.xlsx")

# 指定欄位名稱，計算資料分布比例
label_proportions = df['LABEL'].value_counts(normalize = True)
print("全部資料分布：")
print(label_proportions)

print('--------------')

# 指定要分割的資料標籤欄位，並且要維持資料分布
labels = df['LABEL']

N = 4 # 設定分割數

# 初始化 StratifiedKFold
skf = StratifiedKFold(n_splits = N, shuffle = True, random_state = 42)
# =============================================================================
# 參數說明
# n_splits：指定將數據分成多少份。即交叉驗證中的折數。例如，n_splits = 3 將數據分成三個部分，每次用其中一部分作為測試集，其餘的作為訓練集。
# 
# shuffle：是否在分割數據前打亂數據。預設為 False，設為 True 可以打亂數據，這在處理隨機樣本時有助於提高模型的泛化能力。
# 
# random_state：隨機數生成的種子，用於確保每次運行代碼時結果的一致性。設置這個參數可以讓每次運行的結果一致，便於實驗重現。
# =============================================================================

# 創建列表儲存分割後的 N 筆資料
split_data = [pd.DataFrame() for _ in range(N)]
split_indices = [None for _ in range(N)]

# 分割資料並保持分布
for i, (train_index, _) in enumerate(skf.split(df, labels)):
    split_data[i] = df.iloc[train_index]
    # 保存對應的原始索引
    split_indices[i] = train_index

# 檢查分割後的訓練集資料及其索引
for i, (split, indices) in enumerate(zip(split_data, split_indices)):
    print(f"訓練集資料 {i+1}：")
    print(split['LABEL'].value_counts(normalize = True))
    
    # 打印每個分割的原始索引
    print(f"訓練集資料 {i+1} 在原始資料中的索引：")
    print(indices)
