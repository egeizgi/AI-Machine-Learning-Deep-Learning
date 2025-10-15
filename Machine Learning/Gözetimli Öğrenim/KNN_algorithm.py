from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
# (1) Veri seti incelemesi
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# (2) Makine Öğrenmesi Modelinin Seçilmesi
# (3) Modelin Train Edilmesi
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 42)

#olceklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#knn mpdeli oluştur ve test et
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) # fit fonksiyonu verimizi kullanarak algoritmayı eğitir

# (4) Sonuçların Değerlendirilmesi

prediction = knn.predict(X_test)

accur = accuracy_score(y_test, prediction)
print(f"Doğruluk {accur}")

conf_matrix = confusion_matrix(y_test, prediction)
print(f"Confusion Matrix\n{conf_matrix}")

# (5) Hiperparametre Ayarlaması
acc_values = []
values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accur = accuracy_score(y_test, y_pred)
    print(accur)
    acc_values.append(accur)
    values.append(k)
    
plt.figure()
plt.plot(values, acc_values, marker="o", linestyle="-")
plt.title("K'ya göre doğruluk")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(values)
plt.grid(True)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 1 * (0.5 - np.random.rand(8))
T = np.linspace(0,5,500)[:,np.newaxis]

for weight in ["uniform", "distance"]:
    
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X,y).predict(T)
    plt.figure()
    plt.scatter(X,y, color="green", label="data")
    plt.plot(T,y_pred, color="blue", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title(f"KNN Regressor weights = {weight}")
plt.tight_layout()





















