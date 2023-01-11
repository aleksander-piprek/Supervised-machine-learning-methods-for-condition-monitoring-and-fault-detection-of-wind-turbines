import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

pd.set_option('display.max_columns', None)
mpl.rcParams['agg.path.chunksize'] = 10000

df = pd.read_csv("Dataset/la-haute-borne-data-2013-2016.csv", sep=';')

# pawel_df = pd.read_csv("Scripts/data.csv")
#%% Deleting corrupted data  
df = df[df.Wind_turbine_name == 'R80721']
needed = df.filter(like='avg').columns
df = df[needed]   
del df["Va1_avg"] 
del df["Va2_avg"]
del df["Pas_avg"]

df.drop(df[df['P_avg'] < 0].index, inplace = True)
df.drop(df[df['Q_avg'] < 0].index, inplace = True)
df.drop(df[df['S_avg'] < 0].index, inplace = True)
df.drop(df[df['Ws_avg'] < 0].index, inplace = True)
df.drop(df[df['Ws1_avg'] < 0].index, inplace = True)
df.drop(df[df['Ws2_avg'] < 0].index, inplace = True)
df.drop(df[df['Gb1t_avg'] < 0].index, inplace = True)
df.drop(df[df['Gb2t_avg'] < 0].index, inplace = True)
df.drop(df[df['Db1t_avg'] < 0].index, inplace = True)
df.drop(df[df['Db2t_avg'] < 0].index, inplace = True)
df.drop(df[df['Rbt_avg'] < 0].index, inplace = True)

x1 = np.linspace(4.7,11.5, 100)
y1 = 260*x1 - 1020
for i, j in zip(x1, y1):
    df = df.drop(df[(df['P_avg'] > j) & (df['Ws_avg'] < i)].index)
    
x2 = np.linspace(5.9,12, 100)
y2 = 262.5*x2 - 1475
for k, g in zip(x2, y2):
    df = df.drop(df[(df['P_avg'] < g) & (df['Ws_avg'] > k)].index)

x3 = np.linspace(12.1,13.5, 50)
y3 = 150*x3 - 150
for k1, g1 in zip(x3, y3):
    df = df.drop(df[(df['P_avg'] < g1) & (df['Ws_avg'] > k1)].index)

x4 = np.linspace(14,17.5, 50)
y4 = 60*x3 + 1200
for k2, g2 in zip(x4, y4):
    df = df.drop(df[(df['P_avg'] < g2) & (df['Ws_avg'] > k2)].index)
    
x5 = np.linspace(4,6.25, 10)
y5 = 60*x5 - 255
for k3, g3 in zip(x5, y5):
    df = df.drop(df[(df['P_avg'] < g3) & (df['Ws_avg'] > k3)].index)

df = df.dropna()
df.reset_index(inplace = True, drop = True)
print(df)
#%% Power curve plot
def power():
    plt.figure(dpi=1200)
    plt.title("R80721 Wind Turbine Power curve")
    plt.xlabel("Wind speed [m/s]")
    plt.ylabel("Power [kW]")   
    # plt.scatter(x = pre_df['Ws_avg'], y = pre_df['P_avg'], s=0.1, alpha=0.6, color='r', label='Before cleaning')
    plt.scatter(x = df['Ws_avg'], y = df['P_avg'], s=0.11, color='b', label='After cleaning')
    plt.legend(loc="best")
    
    # plt.scatter(x = pawel_df['Ws'], y = pawel_df['P'], s=0.1, alpha=0.6, color='r')
    
    # plt.plot(x1, y1, linestyle='solid', color='r')
    # plt.plot(x2, y2, linestyle='solid', color='r')
    # plt.plot(x3, y3, linestyle='solid', color='r')
    # plt.plot(x4, y4, linestyle='solid', color='r')
    # plt.plot(x5, y5, linestyle='solid', color='r')
    
    # plt.axhline(y = 2000, color = 'r', linestyle = '-')
    # plt.axvline(x = 12.3, color = 'r', linestyle = '-')     
power()
#%% Gearbox bearing temperature plot
def gbt():
    plt.figure(dpi=1200)
    plt.title("Gearbox bearing temperature")
    plt.xlabel("Samples")
    plt.ylabel("Temperature")
    plt.plot(df.index, df['Gb1t_avg'], color='b')
gbt()
#%% Generator bearing temperature plot
def dbt():
    plt.figure(dpi=1200)
    plt.title("Generator bearing temperature")
    plt.xlabel("Samples")
    plt.ylabel("Temperature")
    plt.plot(df.index, df['Db1t_avg'], color='b')
dbt()
#%% Rotor bearing temperature plot
def rbt():
    plt.figure(dpi=1200)
    plt.title("Rotor bearing temperature")
    plt.xlabel("Samples")
    plt.ylabel("Temperature")
    plt.plot(df.index, df['Rbt_avg'], color='b')
rbt()