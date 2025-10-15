from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

oli = fetch_olivetti_faces()
"""
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i], cmap = "gray")
    plt.axis("off")
plt.show()
"""
X = oli.data
y = oli.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

ests = []
accurs = []
for i, est in enumerate([5,10,50,75,100]):
    forest = RandomForestClassifier(n_estimators = est, random_state = 42)
    forest.fit(X_train, y_train)
    
    y_pred = forest.predict(X_test)
    
    accur = accuracy_score(y_test, y_pred)
    ests.append(est)
    accurs.append(accur)
#print(f"Accuracy score: {accur}")

plt.figure()
plt.scatter(ests,accurs)
plt.grid(True)

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

california = fetch_california_housing()

X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)

forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("rmse: ", rmse, "\nmse: ", mse) 