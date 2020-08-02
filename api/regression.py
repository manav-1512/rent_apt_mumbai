# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('api/static/data/prop_data_clean1.csv')
X = dataset.iloc[:,:-1 ].values
y = dataset.iloc[:, 16].values

a = np.where(X[:, 0]==X[:,1])
for i in a:
    X[i,1]=0
x = np.where(X[:,0]>30)
for i in x:
    X[i,0],X[i,1] = X[i,1],X[i,0]
x = np.where(X[:,0]>21)
for i in x:
    X[i,0]-=10    

from sklearn.impute import SimpleImputer
missingvalues0 = SimpleImputer(missing_values = 1, strategy = 'constant', verbose = 0, fill_value=0)
X[:, 0:1] = missingvalues0.fit_transform(X[:, 0:1])
missingvalues0 = SimpleImputer(missing_values = 7, strategy = 'constant', verbose = 0, fill_value=0)
X[:, 1:2] = missingvalues0.fit_transform(X[:, 1:2])
missingvalues = SimpleImputer(missing_values = 0, strategy = 'constant', verbose = 0, fill_value=np.nan)
X[:, 0:2] = missingvalues.fit_transform(X[:, 0:2])
missingvaluesm = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
X[:, 0:2] = missingvaluesm.fit_transform(X[:, 0:2])

import statsmodels.api as sm
X = np.append(arr = np.ones((34315,1)).astype(int), values = X , axis = 1)
regressor = sm.OLS(y,X).fit()
print(regressor.summary())