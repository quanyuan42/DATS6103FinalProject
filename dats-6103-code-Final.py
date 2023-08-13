#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
data1['trip_time'] = (data1['ended_at'] - data1['started_at']).dt.seconds 
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

#**New Stuff**
column = ['rideable_type','member_casual']
data_dummy=data2.copy()
for col in column:
    label_encoder = LabelEncoder()
    label_encoder.fit(data_dummy[col].values.reshape(-1,1))
    data_dummy[col] = label_encoder.transform(data_dummy[col].values.reshape(-1,1))

#data_dummy.head()

# %%
# extracting month and day from started_at
data_dummy['month'] = data_dummy['started_at'].dt.month
data_dummy['day'] = data_dummy['started_at'].dt.day
# assigning day of week (0 as Monday)
data_dummy['weekday'] = data_dummy['started_at'].dt.weekday
#%%
# *new stuff*
data3 = data_dummy[['day','weekday','start_lat','start_lng','end_lat','end_lng','trip_time']]
corr = data3.corr()
plt.figure(figsize=(10,5))
#g = sns.heatmap(corr,annot=True,cmap="RdYlGn")
#sns.pairplot(data3)


#%%
# compare casual v member in day
# creating subsets
member = data_dummy[data_dummy['member_casual'] == 1] 
casual = data_dummy[data_dummy['member_casual'] == 0]
plt.figure(figsize = (20, 10))
plt.subplot(1,2,1)
plt.title('member')

sns.histplot(data = member, x = 'day' , bins = 10) 
plt.subplot(1,2,2)
plt.title('casual')
sns.histplot(data = casual, x = 'day' , bins = 10) 

plt.figure(figsize = (20, 10))
plt.subplot(1,2,1)
plt.title('member')

sns.histplot(data = member, x = 'weekday' , bins = 10) 
plt.subplot(1,2,2)
plt.title('casual')
sns.histplot(data = casual, x = 'weekday' , bins = 10) 
# %%
#plucking unwanted cols
final_data = data_dummy.drop(['started_at', 'ended_at'], axis = 1)
final_data.info()

#%%
feature = ['month','day','weekday','start_lat','start_lng','end_lat','end_lng']
data4 = final_data.drop(feature, axis = 1)
corr = data4.corr() 
plt.figure(figsize = (10, 10))
g=sns.heatmap(corr, annot = True, cmap = "RdYlGn")

#%%

#Linear Model
x = final_data.drop(['trip_time', 'month','rideable_type','member_casual'], axis = 1)
y = final_data['trip_time']
s1 = StandardScaler()
s1.fit(y.values.reshape(-1, 1))
y = s1.transform(y.values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)
s = StandardScaler()
s.fit(x_train)
X_train_scale = s.transform(x_train)  
X_test_scale = s.transform(x_test)   
#member - docked

# %%
# normalizaing and splitting
lr = linear_model.LinearRegression()
lr.fit(X_train_scale,y_train) 
y_pred = lr.predict(X_test_scale) 
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
r2 = r2_score(y_test, y_pred) 
print("Test MSE: ",round(mse,4))
print("Test RMSE:  ",round(rmse,4))
print("Test R2 :  ",round(r2,3))
plt.scatter(range(len(y_test)),s1.inverse_transform(y_test),label = 'True')  #绘制真实值和预测值的分布散点图
plt.scatter(range(len(y_test)),s1.inverse_transform(y_pred),label = 'Predict')
plt.legend()
plt.show()

#%%
features_import = pd.DataFrame(x_train.columns, columns=['feature'])
features_import['importance'] = lr.coef_[0]
features_import.sort_values('importance', inplace=True)
plt.barh(features_import['feature'], features_import['importance'], height=0.7) 
for a,b in zip( features_import['importance'],features_import['feature']):
    plt.text(a+0.001, b,'%.3f'%float(a))
plt.show()

#%%
# decision tree
x = final_data.drop(['trip_time', 'month'], axis = 1)
y = final_data['trip_time']
s1 = StandardScaler()
s1.fit(y.values.reshape(-1, 1))
y = s1.transform(y.values.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
s = StandardScaler()
s.fit(x_train)
X_train_scale = s.transform(x_train) 
X_test_scale = s.transform(x_test)  

#%%
dc = DecisionTreeRegressor(max_depth = 6)
dc.fit(x_train,y_train) 
y_pred = dc.predict(x_test)  
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
r2 = r2_score(y_test, y_pred) 
print("Test MSE: ",round(mse,4))
print("Test RMSE:  ",round(rmse,4))
print("Test R2 :  ",round(r2,3))

plt.scatter(range(len(y_test)),s1.inverse_transform(y_test.reshape(-1, 1)),label = 'True')  #绘制真实值和预测值的分布散点图
plt.scatter(range(len(y_test)),s1.inverse_transform(y_pred.reshape(-1, 1)),label = 'Predict')
plt.legend()
plt.show()

#%%
features_import = pd.DataFrame(x_train.columns, columns=['feature'])
features_import['importance'] = dc.feature_importances_
features_import.sort_values('importance', inplace = True)
plt.barh(features_import['feature'], features_import['importance'], height=0.7) 
for a,b in zip( features_import['importance'],features_import['feature']):
    plt.text(a+0.001, b,'%.3f'%float(a))
plt.show()
