import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
print("G.RITHANYA 24BAD132")
df=pd.read_csv(r"C:\Users\HP\Downloads\archive (11)\auto-mpg.csv")
df=df[['horsepower','mpg']]
df.replace('?',np.nan,inplace=True)
df['horsepower']=pd.to_numeric(df['horsepower'],errors='coerce')
df.dropna(inplace=True)
X=df[['horsepower']]
y=df['mpg']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
degrees=[2,3,4]
results={}
plt.figure(figsize=(10,6))
for d in degrees:
    poly=PolynomialFeatures(degree=d)
    X_train_poly=poly.fit_transform(X_train_scaled)
    X_test_poly=poly.transform(X_test_scaled)
    model=LinearRegression()
    model.fit(X_train_poly,y_train)
    y_pred=model.predict(X_test_poly)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,y_pred)
    results[d]=(mse,rmse,r2)
    X_plot=np.linspace(X.min(),X.max(),100).reshape(-1,1)
    X_plot_scaled=scaler.transform(X_plot)
    X_plot_poly=poly.transform(X_plot_scaled)
    y_plot=model.predict(X_plot_poly)
    plt.plot(X_plot,y_plot,label=f'Degree {d}')
plt.scatter(X,y,alpha=0.4)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Curve Fitting")
plt.legend()
plt.show()
print("\nMODEL PERFORMANCE")
for d,(mse,rmse,r2) in results.items():
    print(f"\nDegree {d}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
poly=PolynomialFeatures(degree=4)
X_train_poly=poly.fit_transform(X_train_scaled)
X_test_poly=poly.transform(X_test_scaled)
ridge=Ridge(alpha=1.0)
ridge.fit(X_train_poly,y_train)
ridge_pred=ridge.predict(X_test_poly)
ridge_mse=mean_squared_error(y_test,ridge_pred)
ridge_rmse=np.sqrt(ridge_mse)
ridge_r2=r2_score(y_test,ridge_pred)
print("\nRIDGE REGRESSION DEGREE 4")
print(f"MSE: {ridge_mse:.3f}")
print(f"RMSE: {ridge_rmse:.3f}")
print(f"R2: {ridge_r2:.3f}")
train_errors=[]
test_errors=[]
for d in degrees:
    poly=PolynomialFeatures(degree=d)
    X_train_poly=poly.fit_transform(X_train_scaled)
    X_test_poly=poly.transform(X_test_scaled)
    model=LinearRegression()
    model.fit(X_train_poly,y_train)
    train_pred=model.predict(X_train_poly)
    test_pred=model.predict(X_test_poly)
    train_errors.append(mean_squared_error(y_train,train_pred))
    test_errors.append(mean_squared_error(y_test,test_pred))
plt.plot(degrees,train_errors,label="Training Error")
plt.plot(degrees,test_errors,label="Testing Error")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.title("Training vs Testing Error")
plt.legend()
plt.show()
