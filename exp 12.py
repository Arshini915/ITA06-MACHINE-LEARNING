# Iris Flower Classification using KNN

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Step 4: Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Step 5: Train model
knn.fit(X_train, y_train)

# Step 6: Predict
y_pred = knn.predict(X_test)

# Step 7: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Step 8: Predict new flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(sample)

print("Predicted class:", iris.target_names[prediction][0])
