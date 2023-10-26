import numpy as np
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score


# 读取标签数据
def read_labels(filename):
    with open(filename, 'r') as f:
        labels = [float(line.strip()) for line in f]
    return np.array(labels)


# 读取数据
def read_parameters(filename):
    hjorth_params = np.loadtxt(filename, delimiter='\t', skiprows=1)
    return hjorth_params


# 训练 SVM 模型
def train_svm(X_train, y_train):
    clf = svm.SVR()  # Use SVR for continuous labels
    clf.fit(X_train, y_train)
    return clf


# 评估模型
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


if __name__ == "__main__":
    # 读取标签数据
    labels = read_labels("../data/processed/labels_3.dat")

    # 读取 Hjorth 系数数据
    data = read_parameters("../data/processed/hjorth_parameters.dat")

    # 选择最后40行作为测试集
    X_test = data[-40:]
    y_test = labels[-40:]

    # 使用留一被试交叉验证
    loo = LeaveOneOut()
    mse_total, r2_total = 0.0, 0.0
    for train_index, _ in loo.split(data):
        X_train, y_train = data[train_index], labels[train_index]

        # 训练 SVM 模型
        clf = train_svm(X_train, y_train)

        # 评估模型
        mse, r2 = evaluate_model(clf, X_test, y_test)
        mse_total += mse
        r2_total += r2

    # 计算平均 MSE 和 R-squared
    avg_mse = mse_total / len(data)
    avg_r2 = r2_total / len(data)

    print("Mean Squared Error:", avg_mse)
    print("R-squared:", avg_r2)
