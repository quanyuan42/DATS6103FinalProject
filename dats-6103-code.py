#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('202306-capitalbikeshare-tripdata.csv')
print(len(data))
print(data.dtypes)
#%%
#data.head()
#data.shape
data.isnull().sum()/data.shape[0]*100  # inspect missing
# %%
col_not_use=['start_station_name','start_station_id','end_station_name','end_station_id'] #no need cols
data1=data.drop(col_not_use,axis=1)
data1.head()

data1 = data1.dropna() #clear missing
data1.shape
# %%
data1.dtypes
#data1.info
# %%
