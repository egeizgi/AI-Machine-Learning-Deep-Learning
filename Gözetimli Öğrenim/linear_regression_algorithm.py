from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100,1)
y = 3 + 4 * X + np.random.rand(100,1)


lin = LinearRegression()
lin.fit(X, y)

plt.figure()
plt.scatter(X, y)
plt.plot(X, lin.predict(X), color = "red")
plt.xlabel("X")
plt.ylabel('y')

a0 = lin.coef_

# %%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
diabetes = load_diabetes()

diabetes_X, diabetes_y = load_diabetes(return_X_y = True)

diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

lin_reg = LinearRegression()

lin_reg.fit(diabetes_X_train, diabetes_y_train)

pred = lin_reg.predict(diabetes_X_test)

mse = mean_squared_error(diabetes_y_test, pred)
r2 = r2_score(diabetes_y_test, pred)
print(f"mse: {mse}\nr2: {r2}")

plt.scatter(diabetes_X_test, diabetes_y_test, color = "black")
plt.plot(diabetes_X_test, pred, color = "red")