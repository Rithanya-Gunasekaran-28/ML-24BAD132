import pandas as pd
import numpy as np
filepath=r"C:\Users\HP\Downloads\archive.zip"
data=pd.read_csv(filepath,encoding="Latin")
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())
print(data.isnull())
print(data.columns)