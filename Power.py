import numpy as np
from scipy.signal import welch


# 读取数据
data = np.loadtxt("../data/processed/features_raw.dat")

# 定义频带边界
freq_bands = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Lower Beta': (13, 21),
    'Upper Beta': (21, 30),
    'Gamma': (30, 50)
}

# 初始化一个空数组以存储功率特征
power_features = np.zeros((data.shape[0], 5), dtype=float)

# 计算每个用户的功率特征
for i in range(data.shape[0]):
    user_data = data[i, :].reshape(40, 8064)
    power_user = np.zeros((5, 40), dtype=float)

    for channel in range(40):
        signal = user_data[channel, :]
        f, Pxx = welch(signal, fs=250, nperseg=256)
        for band, (low, high) in freq_bands.items():
            band_power = np.trapz(Pxx[(f >= low) & (f <= high)])
            power_user[list(freq_bands.keys()).index(band), channel] = band_power

    # 每列代表一个频带的功率特征
    power_user /= 8064  # 归一化功率特征

    # 将每个通道的功率特征求和，得到每个用户的功率特征
    power_features[i, :] = np.sum(power_user, axis=1)

# 保存功率特征到文本文件，以浮点数格式，不使用科学计数法
np.savetxt("../data/processed/powerfeatures.dat", power_features, fmt='%f')
