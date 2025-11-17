## -------------------------
## Load and Inspect Data
## -------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("insurance.csv")

print(df.head())
print(df.info())
print(df.describe())


## -------------------------
## Encoding categorical variables
## -------------------------
df_encoded = pd.get_dummies(df, drop_first=True)


## -------------------------
## Exploratory Data Analysis (EDA)
## -------------------------
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

print("\nUnique categories:")
for col in df.select_dtypes('object'):
    print(col, df[col].unique())


## -------------------------
## Correlation Heatmap
## -------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


## -------------------------
## Scatter + Regression Plot
## -------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='bmi', y='charges', data=df, alpha=0.4)
sns.regplot(x='bmi', y='charges', data=df, scatter=False, color='red')
plt.title('BMI vs Insurance Charges')
plt.show()


## -------------------------
## ML Imports
## -------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


## -------------------------
## Train/Test Split (using encoded data)
## -------------------------
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## -------------------------
## Train the Model
## -------------------------
model = LinearRegression()
model.fit(X_train, y_train)


## -------------------------
## Predictions
## -------------------------
y_pred = model.predict(X_test)


## -------------------------
## Visualization — Actual vs Predicted
## -------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.show()


## -------------------------
## Residual Plot
## -------------------------
residuals = y_test - y_pred

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


## -------------------------
## Residual Distribution
## -------------------------
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()


## -------------------------
## Model Evaluation
## -------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

