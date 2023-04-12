# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=11)

# Build a kernel SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy score - Linear Kernel: ', round(accuracy_score(y_test, pred),3))

model = SVC(kernel='poly')
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy score - Poly Kernel: ', round(accuracy_score(y_test, pred),3))

model = SVC(kernel='rbf')
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('Accuracy score - RBF Kernel: ', round(accuracy_score(y_test, pred),3))