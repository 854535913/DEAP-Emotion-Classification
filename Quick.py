import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("----------Training----------")
# 1. 读取数据
data_file = '../data/processed/statisticalfeatures.dat'
label_file = '../data/processed/discretedlabels_2.dat'

data = np.loadtxt(data_file) #小波去掉delimiter
labels = np.loadtxt(label_file)

# 2. 划分训练集和验证集
num_validation_samples = 40 # 最后40行作为验证集(留一被试)
X_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
X_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# 3. 创建SVM分类器并训练
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# 4. 使用验证集评估模型
y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print("Result:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("--------------------" + "\n")