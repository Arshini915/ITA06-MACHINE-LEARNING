# Mobile Price Prediction using Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset
# RAM, Storage, Battery, Camera
X = np.array([
    [4, 64, 4000, 12],
    [6, 128, 4500, 48],
    [8, 128, 5000, 64],
    [3, 32, 3000, 8],
    [12, 256, 6000, 108],
    [6, 64, 4500, 48],
    [8, 256, 5000, 64],
    [4, 128, 4000, 16]
])

# Price in thousands
y = np.array([10, 15, 20, 8, 35, 14, 25, 12])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

print("Predicted Prices:", prediction)

# Test with new mobile
new_mobile = [[8, 128, 5000, 64]]

price = model.predict(new_mobile)

print("Predicted Mobile Price:", price[0], "thousand")
