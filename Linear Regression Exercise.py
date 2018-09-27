"""
E-Commerce Project

This project is for a E-commerce company based in New York that sells clothing online, but also have in-store style and clothing advice sessions.
Customers can enter the store and have personal in-store sessions, then they can go home and order the clothes online
either using the website or the mobile app.

The aim of this project is to use Linear Regression to help align the companies business focus.

This project works with the Ecommerce Customers csv file. this csv file contains information such as Customer info,
Email, Address, and their color Avatar as well as numerical information such as:
* Average session length - the average length of in store advice sessions
* Time on App - The average time each client spends on the app
* Time on Website - The average time each client spends on the website
* Length of membership - The duration the client has been a member of our product
* Yearly amount spent - Amount each client has spent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Ecommerce Customers')
print(df.head())
print(df.info())
print(df.describe())
print('Our data has 8 columns - 3 of them categorical data, 5 of them numeric with 500 entries in each column')

sns.set_style('darkgrid')
sns.jointplot(x = df['Time on Website'], y= df['Yearly Amount Spent'])
plt.title('fig.1')
plt.show()
sns.jointplot(data = df, x = 'Time on App', y ='Yearly Amount Spent')
plt.title('fig.2')
plt.show()

print(' ')
print('After reviewing the two plots in fig.1 and fig.2 - fig.1 being Time on Website vs Yearly amount spent and  fig.2'
      ' being Time on App vs Yearly Amount Spent'
      ' We can see that there is a slightly stronger correlation between Sales and time spent on the App. as opposed to'
      ' sales and time spent on website')

sns.jointplot(df['Time on App'], df['Length of Membership'],kind = 'hex')
plt.title('fig.3')
plt.show()
sns.pairplot(df)
plt.title('fig.4')
plt.show()

print(' ')
print("Based of the pairplot(fig.4) - the categories with the strongest linear correlation is Length of Membership and Yearly"
      "Amount Spent, which makes a lot of sense as clients who have longer memberships with a company would purchase"
      "more products")

sns.lmplot('Length of Membership','Yearly Amount Spent',df)
plt.title('fig.5')
plt.show()

print(' ')
print('Based on fig.5 we can already see that there is a decent linear fit between Length of Membership and Yearly Amount spent '
      'as there is little space in between error bars in the graph. '
      'After exploring the data we will build the machine learning model')

X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.3)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
print('prediction results: ', predictions)

plt.scatter(y_test,predictions)
plt.title('fig.6')
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')
plt.show()
print('this scatter plot (fig.6) shows that our model fits very well considering that the data is almost linear')

from sklearn import metrics
print(' ')
print('Model Evaluation')
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(' ')
print('Co-efficient of determination', metrics.explained_variance_score(y_test, predictions) )

sns.distplot(y_test - predictions, bins=50)
plt.show()
print("")
print('This is a histogram of the residuals - as we can see our data is for the most part normally distributed indicating a good fit in our model')

print(' ')
print("coefficients: ", lm.coef_)
print("Intercept: ", lm.intercept_)

cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])
print(cdf)
print("")
print("What this table represents is that for every change in one unit of our features or X, our Y or label increases by the coefficient amount")

print("")
print('After reviewing all data availabble, it\'s evident that the website needs improvement. We can see there is no correlation between'
      ' how much time a client spends on the website and Yearly Amount Spent - indicating that the website does not impact sales. At this point we can'
      ' decide if we want to drop using the website completely and focus more on the digital app, or upgrade the website so that the website'
      ' can become more profitable. It\'s also noted that there is a significant'
      ' correlation between Length of Membership and Yearly Amount Spent, meaning that a promotion that will increase membership lengths'
      ' will have a direct impact on sales')