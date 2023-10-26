import numpy as np
import pywt
import pywt.data

# 读取数据
data_path = '../data/processed/features_raw.dat'
output_path = '../data/processed/waveletfeatures.dat'

# 小波变换和特征提取
def compute_wavelet_features(data):
    features = []
    for row in data:
        coeffs = pywt.wavedec(row, 'db4', level=4)  # 使用四阶Daubechies小波变换
        features.append(np.concatenate(coeffs))
    return np.array(features)


data = np.loadtxt(data_path)

wavelet_features = compute_wavelet_features(data)

# 保存结果到文件
np.savetxt(output_path, wavelet_features)

print(f'Wavelet features saved to {output_path}')
