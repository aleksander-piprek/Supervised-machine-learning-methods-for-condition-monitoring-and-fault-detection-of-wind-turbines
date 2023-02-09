import scipy.io
import pandas as pd

mat = scipy.io.loadmat('Gv.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
data.to_csv("Gv.csv")

