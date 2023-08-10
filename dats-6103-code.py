#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %%
data = pd.read_csv('202306-capitalbikeshare-tripdata.csv')
print(len(data))
print(data.dtypes)
#%%
data.head()
#data.shape
#data.isnull().sum()/data.shape[0]*100  # inspect missing
# %%
col_not_use = ['start_station_name','start_station_id','end_station_name','end_station_id'] #no need cols
data1 = data.drop(col_not_use,axis=1)
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
corr = data3.corr() # correlation matrix, computes the pairwise correlation of columns
plt.figure(figsize=(20, 20))
#g = sns.heatmap(corr,annot = True, cmap = "RdYlGn") 
#sns.pairplot(data3)

# %%
data_dummy.head()
#%%
# compare casual v member
member = data_dummy[data_dummy['member_casual_member'] == 1] 
casual = data_dummy[data_dummy['member_casual_casual'] == 1]
plt.figure(figsize = (20, 20))
plt.subplot(1,2,1)
plt.title('member')

sns.histplot(data = member, x = 'day' , bins = 10) 
plt.subplot(1,2,2)
plt.title('casual')
sns.histplot(data = casual, x = 'day' , bins = 10) 
# %%
#plucking unwanted cols
final_data = data_dummy.drop(['started_at', 'ended_at'], axis = 1)
final_data.head()

# %%
h = ['month','day','weekday','start_lat','start_lng','end_lat','end_lng']
data4 = final_data.drop(h, axis = 1)
corr = data4.corr() # calc correlation
plt.figure(figsize = (20, 20))
#g = sns.heatmap(corr, annot = True, cmap = "RdYlGn")
#member - docked
# %%
# extracting trip_time as y
# normalizaing and splitting
y = final_data['trip_time']
x = final_data.drop(['trip_time', 'month'], axis = 1)
# standardize y
s1 = StandardScaler()
s1.fit(y.values.reshape(-1, 1)) # computes mean and standard deviation for scaling
#y reshaped because it's a 1D array, but fit requires 2D 
y = s1.transform(y.values.reshape(-1, 1))

# split the feature data x and target value y into training and test sets
# set ratio 0.2,  random state to 7.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)

# standardize x_train
s2 = StandardScaler()
s2.fit(x_train)
X_train_scale = s2.transform(x_train) 
X_test_scale = s2.transform(x_test)   

#%%
lr = linear_model.LinearRegression()
lr.fit(X_train_scale, y_train)
y_pred = lr.predict(X_test_scale)  #model prediction
mse1 = mean_squared_error(y_test, y_pred) # calc mse
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred)) #rmse
r2 = r2_score(y_test, y_pred) 
print("Test MSE: ", round(mse1, 4))
print("Test RMSE:  ", round(rmse1, 4))
print("Test R2 :  ", round(r2,3))
plt.scatter(range(len(y_test)),s1.inverse_transform(y_test),label = 'True')  # true v predict
plt.scatter(range(len(y_test)),s1.inverse_transform(y_pred),label = 'Predict')
plt.legend()
plt.show()


# %%
