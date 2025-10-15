import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


X = np.random.rand(100,1) * 4
y = 2 + 3*X ** 2

#plt.scatter(X, y)

poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)

linear = LinearRegression()
linear.fit(X_poly, y)

plt.scatter(X, y, color = 'blue')

X_test = np.linspace(0, 4, 100).reshape(-1, 1)

X_test_poly = poly.transform(X_test)
y_pred = linear.predict(X_test_poly)

plt.plot(X_test, y_pred, color = "red")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polinom Regresyon Modeli')