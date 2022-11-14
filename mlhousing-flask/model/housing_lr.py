# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:36:07 2022

@author: IBRAHIM MUSTAPHA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Importing Dataset
df = pd.read_csv('real-estate.csv')

#Understanding the data
# To fill some missing values?
df = df.fillna(method='ffill')

# Shape of our dataset
#df.shape
print('dimension of housing data: {}'.format(df.shape))

# Info our dataset
df.info()

# Describe our dataset
df.describe()

print(df.head())


#Renaming some columns
df.columns = ['TransDate', 'HouseAge', 'DistoMRT', 'Stores', 'Latitude', 'Longitude', 'HousePrice']

print(df.head())
#Exploratory Data Analysis
df.hist(bins=50, figsize=(15,15))
plt.show()

#let’s find out the correlation of our entire dataset and
#find out which one is most related to median house value.
corr_matrix = df.corr()
corr_matrix['HousePrice'].sort_values(ascending=False)
print(corr_matrix)

#Predictive target
#y = df['HousePrice']

#Exploratory Data Analysis

sns.set(font_scale=1.15)
plt.figure(figsize=(8,4))
sns.heatmap(
   df.corr(),        
    cmap='RdBu_r', 
    annot=True, 
    vmin=-1, vmax=1);
sns.jointplot(x='TransDate',y='HousePrice',data=df)
sns.jointplot(x='HouseAge',y='HousePrice',data=df)
sns.pairplot(df)

         
"""
From the plot, the most promising variable for predicting the House Value is the TransDate
After pre-processing the data, these two columns are not needed anymore: “Latitude, Longitude”, 
so, we drop them from our analysis.
"""
df.drop(['Latitude', 'Longitude'],  axis=1, inplace=True)
df = df[['TransDate', 'HouseAge', 'DistoMRT', 'Stores', 'HousePrice']]
#print(df.head())

#Features from the dataframe
X = df[['TransDate', 'HouseAge', 'DistoMRT', 'Stores']]

#Predictive target
y = df['HousePrice']

  
#splittting the data into Train and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

#Feature Scaling with using those estimated parameters (mean & standard deviation)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#Fitting using multiple linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_std, y_train)

#Model Coefficients for the linear regression
modelcoef = lr.coef_
print('Coefficients: ', modelcoef)
#Model Intercept
intr =lr.intercept_
print('Intercept: ', intr)

# Prediction and testing linear regression
y_pred_lr=lr.predict(X_test_std)

# Report and Accuracy Score
lrscore1 = lr.score(X_train_std, y_train)
lrscore2 = lr.score(X_test_std, y_test)
print('R² of Linear Regression on training set: {:.3f}'.format(lrscore1))
print('R² of Linear Regression on test set: {:.3f}'.format(lrscore2))


#Fitting using Random forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=70)
rf.fit(X_train_std, y_train)


# Prediction and testing random forest
y_pred_rf=rf.predict(X_test_std)

# Report and Accuracy Score
rfscore1 = rf.score(X_train_std, y_train)
rfscore2 = rf.score(X_test_std, y_test)
print('R² of Random Forest Regressor on training set: {:.3f}'.format(rfscore1))
print('R² of Random Forest Regressor on test set: {:.3f}'.format(rfscore2))



# Saving model
pickle.dump(rf, open('estate_forest.pkl', 'wb'))
# Loading model
loaded_model = pickle.load(open('estate_forest.pkl', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


"""
#Calculating house price
x1, x2, x3, x4, x5, x6  = modelcoef

def house_price(var1, var2, var3, var4, var5, var6):
    pred_price = intr + var1 * x1 + var2 * x2 + var3 * x3 + var4 * x4 + var5 * x5 + var6 * x6 
    return pred_price
  
a, b, c, d, e, f =[2013.333, 6.3, 90.45606, 9, 24.97433, 121.5431]

pred_price=house_price(a, b, c, d, e, f)
print(model.predict(a, b, c, d, e, f))

print("Predicted House Price is : " , pred_price)
"""

"""
coeffecients = pd.DataFrame(model.coef_,X.columns)
coeffecients.columns = ['Coeffecients']
coeffecients
"""
