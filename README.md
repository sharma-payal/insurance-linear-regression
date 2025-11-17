📊 Insurance Charges Prediction — Linear Regression Model

This project explores medical insurance cost prediction using Linear Regression.
It includes full EDA, feature engineering, regression modeling, visualizations, and model evaluation on the Insurance dataset.
The goal is to understand how factors like age, BMI, smoking habits, and region influence insurance charges — and to build a predictive model.

🚀 Project Features
✔ Exploratory Data Analysis (EDA)
✔ One-hot encoding for categorical features
✔ Correlation heatmap
✔ Regression line visualization
✔ Train/test split
✔ Linear Regression Model
✔ Error evaluation: MAE, MSE, RMSE, R²
✔ Residual & error analysis plots

📂 Dataset
Dataset: Insurance Charges Dataset
Source: Kaggle
Rows: 1338
Features include:
- age
- sex
- bmi
- children
- smoker
- region
- charges (target)

📈 Key Insights from EDA

🔹 1. Correlation Heatmap
- Smoking has the highest positive correlation with insurance charges.
- BMI and age also significantly influence cost.
- Gender and region show little correlation.

🔹 2. BMI vs Charges (Regression Plot)
- Clear upward trend: higher BMI → higher charges.
- Smokers with high BMI create strong outliers, causing the slope to increase.

🔹 3. Actual vs Predicted Plot
- Points cluster around the diagonal — meaning model fits reasonably well.
- Some deviation at higher charges due to extreme values (smokers with very high fees).

🔹 4. Residual Plot
- Random scatter around zero line → no major pattern, indicating a decent linear fit.
- Larger residuals for very high charge values (expected with this dataset).

📏 Model Performance
Metric	Value
- MAE	4181.19
- MSE	33,596,915.85
- RMSE	5796.28
- R² Score	0.7836

Interpretation:
- The model explains 78.36% of the variance in insurance charges.
- Average prediction error is about $4,181, which is acceptable for real-world medical charges.
- Error increases for extreme high-cost cases — expected in linear regression.

🧠 Technologies Used
- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- NumPy

🔧 How to Run
- git clone https://github.com/sharma-payal/insurance-linear-regression
- cd insurance-linear-regression
- python main.py
