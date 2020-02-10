# Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('fivethirtyeight')

# set epochs number

epochs_nr = 100

# Get freeHK dataset
df_freeHK = pd.read_csv(filepath_or_buffer="C:/Users/Laure/Desktop/freeHK_dataset.csv", parse_dates=['date'])

# show dataset
print(df_freeHK.dtypes)

print(df_freeHK.shape)

print(df_freeHK)

# prediction activity
##########################

# Visualize dataset
plt.figure(figsize=(16, 8))
plt.title('freeHongKong data activity')
plt.plot(df_freeHK['activity'])
plt.gcf().autofmt_xdate()
plt.show()


# create dataframe
data = df_freeHK.filter(['activity'])
dataset = data.values
# number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.79)
print(training_data_len)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# create trainingset
train_data = scaled_data[0:training_data_len+1, :]
x_train = []
y_train = []
for i in range(7, len(train_data)):
    x_train.append(train_data[i-7:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 8:
        print(x_train)
        print(y_train)
        print()

# print(x_train)
# print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=80)

# create testdataset
test_data = scaled_data[training_data_len - 7:, :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(7, len(test_data)):
    x_test.append(test_data[i-7:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('freeHongKong prediction activity ')
plt.xlabel('days', fontsize=18)
plt.ylabel('activity', fontsize=18)
plt.plot(train['activity'])
plt.plot(valid[['activity', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()

# prediction complexity
##########################

# Visualize dataset
plt.figure(figsize=(16, 8))
plt.title('freeHongKong data complexity')
plt.plot(df_freeHK['complexity'])
plt.gcf().autofmt_xdate()
plt.show()


# create dataframe
data = df_freeHK.filter(['complexity'])
dataset = data.values
# number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# create trainingset
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(7, len(train_data)):
    x_train.append(train_data[i-7:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 8:
        print(x_train)
        print(y_train)
        print()

# print(x_train)
# print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=150)

# create testdataset
test_data = scaled_data[training_data_len - 7:, :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(7, len(test_data)):
    x_test.append(test_data[i-7:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('freeHongKong prediction complexity')
plt.xlabel('days', fontsize=18)
plt.ylabel('complexity', fontsize=18)
plt.plot(train['complexity'])
plt.plot(valid[['complexity', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()


# prediction emotionality
##########################

# Visualize dataset
plt.figure(figsize=(16, 8))
plt.title('freeHongKong data emotionality')
plt.plot(df_freeHK['emotionality'])
plt.gcf().autofmt_xdate()
plt.show()

# create dataframe
data = df_freeHK.filter(['emotionality'])
dataset = data.values
# number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# create trainingset
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(7, len(train_data)):
    x_train.append(train_data[i-7:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 8:
        print(x_train)
        print(y_train)
        print()

# print(x_train)
# print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=150)

# create testdataset
test_data = scaled_data[training_data_len - 7:, :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(7, len(test_data)):
    x_test.append(test_data[i-7:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('freeHongKong prediction emotionality ')
plt.xlabel('days', fontsize=18)
plt.ylabel('emotionality', fontsize=18)
plt.plot(train['emotionality'])
plt.plot(valid[['emotionality', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()


# prediction sentiment
##########################

# Visualize dataset
plt.figure(figsize=(16, 8))
plt.title('freeHongKong data sentiment')
plt.plot(df_freeHK['sentiment'])
plt.gcf().autofmt_xdate()
plt.show()

# create dataframe
data = df_freeHK.filter(['sentiment'])
dataset = data.values
# number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# create trainingset
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(7, len(train_data)):
    x_train.append(train_data[i-7:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 8:
        print(x_train)
        print(y_train)
        print()

# print(x_train)
# print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=175)

# create testdataset
test_data = scaled_data[training_data_len - 7:, :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(7, len(test_data)):
    x_test.append(test_data[i-7:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('freeHongKong prediction sentiment ')
plt.xlabel('days', fontsize=18)
plt.ylabel('sentiment', fontsize=18)
plt.plot(train['sentiment'])
plt.plot(valid[['sentiment', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
plt.show()
