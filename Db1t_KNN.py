import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.model_selection import GridSearchCV
mpl.rcParams['agg.path.chunksize'] = 10000
df = pd.read_csv("clean.csv")
#%% Train-test split
df = df.head(100000)
X = df.drop('Db1t_avg',axis=1)
y = df['Db1t_avg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)
#%% Data normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%% KNN
parameters = {'weights' : ['uniform', 'distance'], 
              'n_neighbors' : [12, 10, 8, 7, 5]}
knn = KNeighborsRegressor()
clf = GridSearchCV(knn, parameters)
clf.fit(X_train, y_train)
print(clf.best_params_) 
y_pred = clf.predict(X_test)
print("Generator bearing temperature KNN parameters:")
print(f'R2 score: {r2_score(y_test, y_pred)}')
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_pred)}')
print(f'Max error: {max_error(y_test, y_pred)}')
#%% Plotting
plt.figure(dpi=1200)
# plt.title("Generator bearing temperature KNN")
plt.xlabel("Samples")
plt.ylabel("Temperature [°C]")  
plt.plot(np.arange(y_test.size), y_test, color = 'b', label='True') 
plt.plot(np.arange(y_pred.size), y_pred, color = 'r', label='Predicted')
plt.legend(loc="best")
plt.show()