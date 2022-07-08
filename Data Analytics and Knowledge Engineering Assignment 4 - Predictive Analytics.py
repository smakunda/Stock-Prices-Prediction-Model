#!/usr/bin/env python
# coding: utf-8

# # ADDIKAH SYDNEY MAKUNDA
# # 21/03313
# # Assignment four - Predictive Analytics
# # Predicting IBM stock prices
# # Data source: kaggle
# # Importing The Necessary Libraries and Loading The Data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
#from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error


# In[2]:


# Necessary code that will aid in data visualization
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real IBM Stock Price')
    plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


# In[19]:


data=pd.read_csv('C:/Datafiles/AABA_File.csv', index_col='Date', parse_dates=['Date'])
print('The shape of the dataframe is :\n',data.shape)
print(data)


# ### Selecting training and test data. Selected 2006 to 2016 data range as training data and 2017 and beyond as test data

# In[4]:


training_set = data[:'2016'].iloc[:,1:2].values
test_set = data['2017':].iloc[:,1:2].values


# ### Test with "High" price attribute and display the trend over the years

# In[5]:


data["High"][:'2016'].plot(figsize=(16,4),legend=True)
data["High"]['2017':].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])
plt.title('IBM stock price')
plt.show()


# ## Preprocess the train and test data

# In[6]:


# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[7]:


# create a data structure with 60 timesteps and 1 output since LSTMs store long term memory state
# So for each element of training set, we have 60 previous training set elements 
X_train = []
y_train = []
for i in range(60,2768):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[8]:


# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))


# ## Create LSTM architecture by creating layers, compiling the RNN and training the data set

# In[9]:


# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=50,batch_size=32)


# ### Create array to hold stock

# In[11]:


# Get Test set ready same way as the training set.
# The following has been done so first 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 
# 'High' attribute data for processing
data_total = pd.concat((data["High"][:'2016'],data["High"]['2017':]),axis=0)
inputs = data_total[len(data_total)-len(test_set) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)


# In[12]:


# Preparing X_test and predicting the prices
X_test = []
for i in range(60,311):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[13]:


# Visualizing the results for LSTM
plot_predictions(test_set,predicted_stock_price)


# In[14]:


# Evaluating our model
return_rmse(test_set,predicted_stock_price)


# ### The LSTM model architecture has a root mean squared error of 4.724768468613017 as shown above. The error size is somehow big as can be shown by the gap in the trend chart

# # Create GRU architecture by creating layers, compiling the RNN and predicting the prices

# In[15]:


# The GRU architecture
regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
# Fitting to the training set
regressorGRU.fit(X_train,y_train,epochs=50,batch_size=150)


# In[16]:


# Preparing X_test and predicting the prices
X_test = []
for i in range(60,311):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
GRU_predicted_stock_price = regressorGRU.predict(X_test)
GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)


# In[17]:


# Visualizing the results for GRU
plot_predictions(test_set,GRU_predicted_stock_price)


# In[18]:


# Evaluating GRU
return_rmse(test_set,GRU_predicted_stock_price)


# ### The GRU model architecture has a root mean squared error of 2.383560688346273 as shown above. The error size is less than that for LSTM and in this case gives a better prediction as can be seen in the trendline graph
