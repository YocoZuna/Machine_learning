#importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.impute import SimpleImputer
#importing data set
dataset = pd.read_csv("Data.csv")
# Creating vectors X and Y where Y is predicted value X input values   
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

# Taking care of missing data by replacing NaN values with mean of all values in column
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
# Calculating the mean 
imputer.fit(X[:,1:3])
# Adding values to NaN rows
X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding critrical data 
#One hot encoding changing colum with countrys into 3 columns with 3 binary values 
#so there is no corraltions betwen country example. France-0 Germany-1 -> Thas is bad 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X)) 

#Encoding dependent variables 
# changing yes  and no into 1 and 0 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)

# Spliting data into test and traning set 

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=1)

## Feature scaling 
# We can do normalisation and standardisation
# Normalization (X-min(x))/max(X)-min(x)
#Standardisation (X-mean(x))/snadard deviation(x)
# Standardisation is used more offten 
# We do this on both featurs matrix -> X_train and X_test 
 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
# We dont want feater scale first 3 colums beacuse they correspond to the country  
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
# We use only transform to use the same mean and std for  test set 
# Using fit will change te mean and std value which will affect the result 
X_test[:,3:] = sc.transform(X_test[:,3:])


