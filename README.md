# Coincourse19-20-LSTM-Condor-timeseries-prediction
Coincourse19-20-LSTM-Condor-timeseries-prediction

LSTM prediciton of the condor timeseries based on the stock prediciton https://github.com/randerson112358/Python/blob/master/LSTM_Stock/LSTM2.ipynb by randerson112358

the following importes are needed

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
