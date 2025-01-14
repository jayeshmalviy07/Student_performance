import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import zipfile
import requests
import io

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    file_name = z.namelist()[0]  # Get the first file in the zip archive
    with z.open(file_name) as f:
        students_data = pd.read_csv(f, sep=';')

print("Dataset Shape:", students_data.shape)
print("\nDataset Head:\n", students_data.head())

students_data = pd.get_dummies(students_data, drop_first=True)

X = students_data.drop("G3", axis=1)
y = students_data["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_preds = linear_model.predict(X_test)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Model Evaluation
print("\nLinear Regression Metrics:")
print("MAE:", mean_absolute_error(y_test, linear_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_preds)))
print("R2 Score:", r2_score(y_test, linear_preds))

print("\nRandom Forest Metrics:")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("R2 Score:", r2_score(y_test, rf_preds))

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=rf_preds, label="Predictions vs Actuals")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label="Ideal Fit")
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Random Forest Predictions")
plt.legend()
plt.show()

