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
data.head()
#data.shape
#data.isnull().sum()/data.shape[0]*100  # inspect missing
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
# converting time df
data1['started_at']=pd.to_datetime(data1['started_at'])
data1['ended_at']=pd.to_datetime(data1['ended_at'])
#print(data1['started_at'])
# %%
# calc trip time (in seconds)
data1['trip_time']=(data1['ended_at']-data1['started_at']).dt.seconds 
data1.head()

# %%
#forgot to pluck ride id out
data2 = data1.drop(['ride_id'],axis=1)

#%%
x1 = data2['member_casual'].value_counts()
x2 = x1.index
y1 = [x1['member'], x1['casual']]
# pie chart of member v casual (non-member)
plt.pie(y1, labels=x2, autopct='%1.1f%%') # to show in percentage
plt.show()

# %%
#one-hot encoding
column = ['rideable_type','member_casual']
data_dummy = pd.get_dummies(data2, columns = column, prefix = column, prefix_sep = "_") #convertig into dummy variables
data_dummy.head()
# %%
# extracting month and day from started_at
data_dummy['month'] = data_dummy['started_at'].dt.month
data_dummy['day'] = data_dummy['started_at'].dt.day
# assigning day of week (0 as Monday)
data_dummy['weekday'] = data_dummy['started_at'].dt.weekday
#%%
# creating heatmap
data3 = data_dummy[['day','weekday','start_lat','start_lng','end_lat','end_lng','trip_time']]
corr= data3.corr() # correlation matrix, computes the pairwise correlation of columns
#plt.figure(figsize=(20, 20))
#g = sns.heatmap(corr,annot = True, cmap = "RdYlGn") 
#sns.pairplot(data3)

