import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

print("RITHANYA.G 24BAD132 EXP-3 SCENARIO-1")

path=r"C:\Users\HP\Downloads\archive (9)\StudentsPerformance.csv"
data = pd.read_csv(path)
print(data.head())


le = LabelEncoder()

data['parental level of education'] = le.fit_transform(
    data['parental level of education']
)

data['test preparation course'] = le.fit_transform(
    data['test preparation course']
)


#Target variable
data['final_score'] = (
    data['math score'] +
    data['reading score'] +
    data['writing score']
) / 3

np.random.seed(1)

data['study_hours'] = np.random.randint(1, 6, len(data))
data['attendance'] = np.random.randint(60, 100, len(data))
data['sleep_hours'] = np.random.randint(5, 9, len(data))

#Input Features
X = data[['study_hours',
          'attendance',
          'sleep_hours',
          'parental level of education',
          'test preparation course']]

y = data['final_score']

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

for name, coef in zip(X.columns, model.coef_):
    print(name, ":", coef)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Predicted vs Actual Scores")
plt.show()

plt.figure()
plt.bar(X.columns, model.coef_)
plt.xlabel("Input Features")
plt.ylabel("Coefficient Value")
plt.title("Coefficient Magnitude Comparison")
plt.xticks(rotation=45)
plt.show()

# Residual Distribution Plot

residuals = y_test - y_pred

plt.figure()
plt.hist(residuals, bins=20)
plt.xlabel("Residuals (Actual − Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution Plot")
plt.show()

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

print("Ridge R2:", ridge.score(X_test, y_test))
print("Lasso R2:", lasso.score(X_test, y_test))