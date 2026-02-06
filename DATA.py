 import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path=r"C:\Users\HP\Downloads\archive (4).zip"
my_data=pd.read_csv(file_path,encoding="latin1",sep="\t")
print("RITHANYA.G \n24BAD132\n")
print(my_data.head())
print(my_data.tail())
print(my_data.info())
print(my_data.describe())
print(my_data.isnull())
print(my_data.columns)
my_data['Age'] = 2026 - my_data['Year_Birth']
# BAR PLOTS
# Age bar plot
plt.figure(figsize=(6,4))
sns.barplot(x=my_data['Age'].value_counts().index,
            y=my_data['Age'].value_counts().values)
plt.title("Age Distribution")
plt.show()
# Income bar plot
plt.figure(figsize=(6,4))
sns.barplot(x=my_data['Income'].value_counts().index[:10],
            y=my_data['Income'].value_counts().values[:10])
plt.title("Income Distribution")
plt.show()
# Spending bar plot (Wine)
plt.figure(figsize=(6,4))
sns.barplot(x=my_data['MntWines'].value_counts().index[:10],
            y=my_data['MntWines'].value_counts().values[:10])
plt.title("Spending Pattern (Wine)")
plt.show()
# BOX PLOTS
# Age box plot
plt.figure(figsize=(5,3))
sns.boxplot(x=my_data['Age'])
plt.title("Age Box Plot")
plt.show()
# Income bar plot
plt.figure(figsize=(5,3))
sns.boxplot(x=my_data['Income'])
plt.title("Income Box Plot")
plt.show()
# Spending bar plot (Wine)
plt.figure(figsize=(5,3))
sns.boxplot(x=my_data['MntWines'])
plt.title("Spending Box Plot")
plt.show()
