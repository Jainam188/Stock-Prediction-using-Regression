import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime


df = pd.read_csv('RELIANCE.NS.csv')
df.dropna(inplace=True)

print(data.head())
print(data.info())


df = df.reset_index()
prices = df['Close'].tolist()
dates = df.index.tolist()

# prices = df['Close']
# dates = df['Date']
# df.set_index(dates, inplace=True)

#Convert to 1d Vector
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices), 1))

# data = df.sort_index(ascending=True, axis=0)
# new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
#
# for i in range(0, len(data)):
#     new_data['Date'][i] = data['Date'][i]
#     new_data['Close'][i] = data['Close'][i]
#
# new_data['Date'] = pd.to_datetime(new_data['Date'], format='%Y/%m/%d')
#
# new_data.set_index(new_data['Date'], inplace=True)
# new_data.dropna(inplace=True)
#
# print(new_data.head())

regressor = LinearRegression()

# #Splitting the dataset into the Training set and Test set
xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.25, random_state=42)

#Fitting The Model
regressor.fit(xtrain, ytrain)

# pred = regressor.predict(xtrain)
preds = regressor.predict(xtest)

#Finding The Error and Checking The Score
rms = np.sqrt(np.mean(np.power((np.array(ytest)-np.array(preds)), 2)))
print(rms)
print('Score of the Linear Regression Model', regressor.score(xtest, ytest))

#plotting The Prediction With Actual Value
plt.scatter(xtrain, ytrain, color = 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title('Actual Price VS Predicted Price')
plt.show()
