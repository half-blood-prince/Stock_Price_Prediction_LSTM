#Description: this program uses an rnn called lstm
#             to pridict the closing the stock price of a corporation. using the past 60day stock price

#import the library
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#get the stock quote
df= web.DataReader('AAPL',data_source='yahoo',start='2012-01-01',end='2021-12-17')
print(df.head())
#print(df)

pip install --upgrade pandas DataReader

#get the number of rows and columns in the dataset
df.shape

#visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('close price history')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=24)
plt.ylabel('close price USD($)',fontsize=24)
plt.show()


#create a new dataframe with only close column
data=df.filter(['Close'])
#convert the dataframe to a numpy array
dataset=data.values
#get the number of rows to train the model on
train_data_len=math.ceil(len(dataset)*.8)
print(train_data_len)
dataset



#scale the data
scaler =MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data



#create the training data set
#create the scaled training data set
train_data=scaled_data[0:train_data_len, :]
#split the data into x_train and y_train data sets
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
  if i<=61:
    print(x_train)
    print(y_train)
    print()  
    
    
    
#convert the x_train and y_train to numpy arrays
x_train,y_train=np.array(x_train), np.array(y_train)



#reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

#build the lstm model
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


#compile the model
model.compile(optimizer='adam',loss='mean_squared_error')
