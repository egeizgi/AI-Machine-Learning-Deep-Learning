from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

tree = DecisionTreeClassifier()
tree_param = {'max_depth': [3,5,7]}

#KFOLD Grid Search
kfold = KFold(n_splits=10)
tree_grid_search_kf = GridSearchCV(tree, tree_param, cv=kfold)
tree_grid_search_kf.fit(X_train, y_train)

print("KF Best parameter: ",tree_grid_search_kf.best_params_)
print("KF Best score: ",tree_grid_search_kf.best_score_)

#LOO
loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_param, cv = loo)
tree_grid_search_loo.fit(X_train, y_train)

print("\nLOO Best parameter: ",tree_grid_search_loo.best_params_)
print("LOO Best score: ",tree_grid_search_loo.best_score_)