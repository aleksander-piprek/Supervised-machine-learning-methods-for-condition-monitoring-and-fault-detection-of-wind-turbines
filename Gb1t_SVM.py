import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error

mpl.rcParams['agg.path.chunksize'] = 10000

df = pd.read_csv("clean.csv", sep=',')

#%% Train-test split
df = df.head(10000)
X = df.drop('Gb1t_avg',axis=1)

y = df['Gb1t_avg']
# print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%% SVM
svclassifier = SVR()
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Gearbox bearing temperature:")
print(f'R2 score: {r2_score(y_test, y_pred)}')
print(f'R2 mean_squared_error: {mean_squared_error(y_test, y_pred)}')

plt.figure(dpi=1200)
plt.title("True and predicted values of the gearbox bearing temperature")
plt.xlabel("Samples")
plt.ylabel("Temperature")  
plt.plot(np.arange(y_pred.size), y_pred, color = 'r')
plt.plot(np.arange(y_test.size), y_test, color = 'b')
plt.show()