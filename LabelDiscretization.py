print("----------Label Discretization----------")

with open('data/processed/labels_0.dat', 'r') as file:
    labels = file.read().split()

# 将标签划分为0和1
discreted_labels = [1 if float(label) >= 5 else 0 for label in labels]

# 将结果保存到discretedlabels_0.dat文件
with open('data/processed/discretedlabels_0.dat', 'w') as file:
    file.write('\n'.join(map(str, discreted_labels)))
print("labels_0")

##################################################################
with open('data/processed/labels_1.dat', 'r') as file:
    labels = file.read().split()

discreted_labels = [1 if float(label) >= 5 else 0 for label in labels]

with open('data/processed/discretedlabels_1.dat', 'w') as file:
    file.write('\n'.join(map(str, discreted_labels)))
print("labels_1")
##################################################################
with open('data/processed/labels_2.dat', 'r') as file:
    labels = file.read().split()

discreted_labels = [1 if float(label) >= 5 else 0 for label in labels]

with open('data/processed/discretedlabels_2.dat', 'w') as file:
    file.write('\n'.join(map(str, discreted_labels)))
print("labels_2")
##################################################################
with open('data/processed/labels_3.dat', 'r') as file:
    labels = file.read().split()

discreted_labels = [1 if float(label) >= 5 else 0 for label in labels]

with open('data/processed/discretedlabels_3.dat', 'w') as file:
    file.write('\n'.join(map(str, discreted_labels)))
print("labels_3")
print("--------------------" + "\n")