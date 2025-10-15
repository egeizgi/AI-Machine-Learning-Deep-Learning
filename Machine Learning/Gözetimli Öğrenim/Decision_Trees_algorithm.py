from sklearn import datasets
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# (1) Veri seti inceleme
iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

# (2) Model seçme
tree  = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
tree.fit(X_train,y_train)

# (3) Evaluation test
pred = tree.predict(X_test)
accur = accuracy_score(y_test, pred)
print(f"Iris veri seti accur: {accur}")
conf = confusion_matrix(y_test, pred)
print(f"conf_matrix: \n{conf}")

plt.figure(figsize=(15,10))
plot_tree(tree, filled=True, feature_names= iris.feature_names, class_names=iris.target_names)

feature_impt = tree.feature_importances_

feature_names = iris.feature_names

all_sorted = sorted(zip(feature_impt, feature_names))

for importance, feature_name in all_sorted:
    print(f"{importance} : {feature_name}")
    
    
