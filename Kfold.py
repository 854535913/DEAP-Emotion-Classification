import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

print("----------Training----------")
# 1. 读取数据
data_file = '../data/processed/statisticalfeatures.dat'
label_file = '../data/processed/discretedlabels_0.dat'

data = np.loadtxt(data_file)
labels = np.loadtxt(label_file)

# 2. 随机划分训练集和测试集
train_size = 0.7
test_size = 0.3
num_trials = 1

# 3.初始化模型
clf = svm.SVC(kernel='linear')

for trial in range(num_trials):
    print(f"----------Training - Trial {trial + 1}----------")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=trial)

    #  继续训练，基于前一次的模型
    clf.fit(X_train, y_train)

    #  使用测试集评估模型
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Result:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# 4. K-Fold交叉验证
k = 32
num_subjects = 32
data_per_subject = 40
kf_results = cross_val_score(clf, data, labels, cv=k)

print("K-Fold Cross-Validation Results:")
print("Accuracy: {:.2f}".format(np.mean(kf_results)))
