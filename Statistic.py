import numpy as np
import pandas as pd

print("----------FeatureExtraction-Statistic----------")
# 1. 加载数据
data_path = "../data/processed/features_raw.dat"
output_path = "../data/processed/statisticalfeatures.dat"

data = np.loadtxt(data_path)
# 2. 计算各项统计特征
mean = np.mean(data, axis=1)
std_dev = np.std(data, axis=1)
diff1 = np.mean(np.abs(np.diff(data, axis=1)), axis=1)
normalized_data = (data - np.mean(data, axis=1)[:, np.newaxis]) / np.std(data, axis=1)[:, np.newaxis]
diff1_normalized = np.mean(np.abs(np.diff(normalized_data, axis=1)), axis=1)
diff2 = np.mean(np.abs(np.diff(data, n=2, axis=1)), axis=1)
diff2_normalized = np.mean(np.abs(np.diff(normalized_data, n=2, axis=1)), axis=1)

# 3. 将特征合并为一个数据帧
features_df = pd.DataFrame({
    'Mean': mean,
    'Standard Deviation': std_dev,
    'Mean of Absolute Diff 1': diff1,
    'Mean of Absolute Diff 1 (Normalized EEG)': diff1_normalized,
    'Mean of Absolute Diff 2': diff2,
    'Mean of Absolute Diff 2 (Normalized EEG)': diff2_normalized
})

features_data = features_df.to_numpy()
np.savetxt(output_path, features_data, fmt='%f', delimiter='\t')
print(f'Statistical features saved to {output_path}')
print("--------------------" + "\n")