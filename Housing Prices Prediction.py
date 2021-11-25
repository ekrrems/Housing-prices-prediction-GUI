import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor    
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score

data = pd.read_excel(r'C:\Users\ekrem\OneDrive\Masaüstü\Machine learning DATAS\House Sale Datas.xlsx')
df = data.copy()
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()

print(df.shape)
print(df.isna().sum())
print(df.info())

print(df.describe().T)

X = df.drop('Fiyat',axis=1)
y = df['Fiyat']

print('X Shape: {}\ny Shape: {}'.format(X.shape,y.shape))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
                                                 random_state=42)
model = RandomForestRegressor()

model.fit(X_train,y_train)

#Model Prediction

y_pred = model.predict(X_test)

#Model Accuracy

r2_score(y_test,y_pred)

#Creation of GUI Using Tkinter

import tkinter as tk  
from functools import partial  
   
   
def call_result(label_result, n1, n2,n3,n4):  
    num1 = (n1.get())  
    num2 = (n2.get()) 
    num3 = (n3.get())
    num4 = (n4.get())
    result = model.predict([[int(num1),int(num2),int(num3),int(num4)]])  
    label_result.config(text="House Value:{} K".format(result[0]))  
    return  
   
root = tk.Tk()  
root.geometry('350x150')  
  
root.title('Home Price Prediction')  
   
number1 = tk.StringVar()  
number2 = tk.StringVar()  
number3 = tk.StringVar()
number4 = tk.StringVar()

labelNum1 = tk.Label(root, text="Rooms").grid(row=1, column=0)  
  
labelNum2 = tk.Label(root, text="M2").grid(row=2, column=0)  

labelNum3 = tk.Label(root,text='Story').grid(row=3,column =0)

labelNum4 = tk.Label(root,text='Age').grid(row=4,column=0)
  
labelResult = tk.Label(root)  
  
labelResult.grid(row=7, column=2)  
  
entryNum1 = tk.Entry(root, textvariable=number1).grid(row=1, column=2)  
  
entryNum2 = tk.Entry(root, textvariable=number2).grid(row=2, column=2)  

entryNum3 = tk.Entry(root, textvariable=number3).grid(row=3,column=2)

entryNum4 = tk.Entry(root,textvariable=number4).grid(row=4,column=2)
  
call_result = partial(call_result, labelResult, number1, number2,number3,number4)  
  
buttonCal = tk.Button(root, text="Estimated House Value", command=call_result).grid(row=5, column=1)  
  
root.mainloop()








