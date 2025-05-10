import numpy as np

import MyFuncs as mf
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X_train_hog, y_train, X_test_hog, y_test = mf.load_data()

param_grid = {'C': [0.1, 0.5, 1, 5]}
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train_hog, y_train)

best_C = grid_search.best_params_
model = grid_search.best_estimator_

y_pred = model.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)

support_vectors = model.support_vectors_
support_indices = model.support_  # 支持向量在训练集中的索引
n_support = model.n_support_  # 每个类别的支持向量数量
dual_coef = model.dual_coef_  # 支持向量的系数 (yi * αi)


print(f"最佳C值: {best_C}")
print(f"最佳线性SVM的准确率: {accuracy:.4f}")
print(f"支持向量总数: {len(support_indices)}")
print(f"正类支持向量数: {n_support[1]}")  # 假设正类标签为1
print(f"负类支持向量数: {n_support[0]}")  # 假设负类标签为0

X_train = np.load('X_train_sampled.npy')
n_rows = len(support_indices) // 5 + 1

plt.figure(figsize=(15, n_rows * 4))
for i, idx in enumerate(support_indices):
    plt.subplot(n_rows, 5, i + 1)  # 每行显示5个图像
    plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    plt.title(f"class: {y_train[idx]}\n weights: {abs(dual_coef[0][i]):.4f}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('support_vectors.png')
plt.show()