import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import numpy as np

data = pd.read_csv('RELIANCE.NS.csv')

# print(data.head())
# print(data.max())
print(data.info())
data.set_index('Date', inplace=True)

# print(data.head())

plt.figure(figsize=(16, 10))
plt.plot(data['Close'], label='Close Price History')
plt.legend()
# plt.show()

data['Date'] = pd.to_datetime(data.index, format='%Y/%m/%d')

data1 = data.sort_index(ascending=True, axis=0)

df = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])


for i in range(0, len(data1)):
     df['Date'][i] = data['Date'][i]
     df['Close'][i] = data['Close'][i]

df.set_index('Date', inplace=True)
df.dropna(inplace=True)
print(df.head(20))

train = df[:987]
valid = df[987:]

# from sklearn.model_selection import train_test_split
# xtrain, ytrain, xtest, ytest = train_test_split(data['Date'], data['Close'], test_size=0.25)

x_train = np.array(train.index).reshape(-1, 1)
y_train = np.array(train['Close']).reshape(-1, 1)

x_valid = np.array(valid.index).reshape(-1, 1)
y_valid = np.array(valid['Close']).reshape(-1, 1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)), 2)))
print(rms)

# valid['Predictions'] = 0
# valid['Predictions'] = preds
#
# valid.index = df[987:].index
# train.index = df[:987].index
#
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.show()

# # dates = np.reshape(Dates, (len(Dates), 1))
# # prices = np.reshape(Prices, (len(Prices), 1))
#
# df['Date'] = pd.to_datetime(df.Date, format='%Y/%m/%d')
#
#
#
# print(df.head(15))
#
#
#
# x_train = np.array(train['Date']).reshape(-1, 1)
# y_train = np.array(train['Close'])
#
# x_test = np.array(test['Date']).reshape(-1, 1)
# y_test = np.array(test['Close'])
#
# from sklearn import linear_model
#
# model = linear_model.LinearRegression()
# model.fit(x_train, y_train)
#
# pred = model.predict(x_test)
# # rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(pred)), 2)))
# # print(rms)
