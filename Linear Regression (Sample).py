import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('USA_Housing.csv')

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
  'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.4, random_state = 101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)
print(lm.coef_)

cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
print(cdf)
"""
Basically what these coefficients mean is that if we hold all other units fixed, we change one unit in each column
i.e we change one unit in Avg. Area income, we add on 21.52 in price

this project was used fake data - from sklearn.datasets import load_boston
save df = load_boston()
boston.keys()

"""

### Predictions

predictions = lm.predict(X_test)
print ("Prediction Results:", predictions)

plt.scatter(y_test, predictions)
plt.show()
"""
Notice that this chart the data is relatively linear, that means that our estimation for what the data is
is very close to the real data

"""

sns.distplot((y_test-predictions))
plt.show()

from sklearn import metrics
print ('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print ('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print ('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


