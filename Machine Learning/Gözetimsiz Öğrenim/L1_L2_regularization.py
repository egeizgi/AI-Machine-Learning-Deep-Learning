from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import root_mean_squared_error
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)


#Ridge
ridge = Ridge()
ridge_params = {'alpha': [0.1, 1, 10, 100]}

ridge_grid_search = GridSearchCV(ridge, ridge_params, cv = 5)
ridge_grid_search.fit(X_train, y_train)

print("Ridge Best Parameters: ", ridge_grid_search.best_params_)
print("Ridge Best Scores: ", ridge_grid_search.best_score_)

best_ridge_model = ridge_grid_search.best_estimator_
ridge_pred = best_ridge_model.predict(X_test)

ridge_rmse = root_mean_squared_error(y_test, ridge_pred)
print("ridge_rmse:",ridge_rmse)

#Lasso
lasso = Lasso()
lasso_params = {'alpha': [0.1, 1, 10, 100]}
lasso_grid_search = GridSearchCV(lasso, lasso_params, cv = 5)

lasso_grid_search.fit(X_train, y_train)

print("\nLasso Best Parameters: ", lasso_grid_search.best_params_)
print("Lasso Best Scores: ", lasso_grid_search.best_score_)