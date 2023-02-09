import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
mpl.rcParams['agg.path.chunksize'] = 10000
df = dd.read_csv("clean.csv")
#%% Train-test split
df = df.head(30000)
X = df.drop('Gb1t_avg',axis=1)
y = df['Gb1t_avg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
#%% Data normalization
scaler = MinMaxScaler()
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)
#%% GPR
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
gpr.fit(X_train, y_train)
y_pred = gpr.predict(X_test)
print("Gearbox bearing temperature GPR parameters:")
print(f'R2 score: {r2_score(y_test, y_pred)}')
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_pred)}')
print(f'Max error: {max_error(y_test, y_pred)}')
#%% Plotting
plt.figure(dpi=1200)
# plt.title("Gearbox bearing temperature GPR")
plt.xlabel("Samples")
plt.ylabel("Temperature [°C]") 
plt.plot(np.arange(y_test.size), y_test, color = 'b', label='True') 
plt.plot(np.arange(y_pred.size), y_pred, color = 'r', label='Predicted')
plt.legend(loc="best")
plt.show()
#%%