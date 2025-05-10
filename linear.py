import MyFuncs as mf
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

X_train_hog, y_train, X_test_hog, y_test = mf.load_data()

param_grid = {'C': [0.1, 0.5, 1, 5]}
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train_hog, y_train)

best_C = grid_search.best_params_
model = grid_search.best_estimator_

y_pred = model.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)

print(f"最佳C值: {best_C}")
print(f"最佳线性SVM的准确率: {accuracy:.4f}")