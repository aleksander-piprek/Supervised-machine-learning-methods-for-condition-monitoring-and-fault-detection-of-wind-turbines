import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.model_selection import GridSearchCV
mpl.rcParams['agg.path.chunksize'] = 10000
df = pd.read_csv("clean.csv")
#%% Train-test split
df = df.head(50000)
X = df.drop('Gb1t_avg',axis=1)
y = df['Gb1t_avg']
# print(y.value_counts(bins=10))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#%% Data normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%% SVM
parameters = {'kernel':('linear', 'rbf'), 
              'C':[1, 10]}
svclassifier = SVR()
clf = GridSearchCV(svclassifier, parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Gearbox bearing temperature SVM parameters:")
print(f'R2 score: {r2_score(y_test, y_pred)}')
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_pred)}')
print(f'Max error: {max_error(y_test, y_pred)}')
#%% Plotting
plt.figure(dpi=1200)
plt.xlabel("Samples")
plt.ylabel("Temperature [°C]") 
plt.plot(np.arange(y_test.size), y_test, color = 'b', label='True') 
plt.plot(np.arange(y_pred.size), y_pred, color = 'r', label='Predicted')
plt.legend(loc="best")
plt.show()
#%%
df = pd.read_csv("clean.csv")
df = df.iloc[43750:53750]
y_test = df['Gb1t_avg']
df = df.drop('Gb1t_avg',axis=1)
X_test = scaler.transform(df)
y_pred = clf.predict(X_test)

plt.figure(dpi=1200)
plt.xlabel("Samples")
plt.ylabel("Temperature [°C]") 
plt.plot(np.arange(y_test.size), y_test, color = 'b', label='True') 
plt.plot(np.arange(y_pred.size), y_pred, color = 'r', label='Predicted')
plt.legend(loc="best")
plt.show()

diff = y_test - y_pred
plt.figure(dpi=1200)
plt.xlabel("Samples")
plt.ylabel("Temperature [°C]") 
plt.plot(np.arange(y_pred.size), diff, color = 'r', label='Difference in temperature')
plt.axhline(y = 0, color = 'b', linestyle = '-', label='Mean')  
plt.axhline(y = 1.25, color = 'm', linestyle = '-', label='95% Confidence')  
plt.axhline(y = -1.25, color = 'm', linestyle = '-')     
plt.axhline(y = 1.75, color = 'y', linestyle = '-', label='99% Confidence')  
plt.axhline(y = -1.75, color = 'y', linestyle = '-')     
plt.legend(loc="best")
plt.show()

