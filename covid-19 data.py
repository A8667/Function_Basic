import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

covid = pd.read_csv('C:\\Users\\Lenovo\\Downloads\\New folder\\covid.csv')
covid.head()

covid =covid[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
covid.columns= ['date','state','cured','deaths','confirmed']
covid.head()
covid.tail()

today = covid[covid.date=='08-05-2021']
today.head(10)
today.shape

max_confirmed_case = today.sort_values('confirmed',ascending=False)
top_confirmed_case= max_confirmed_case.head(5)
top_confirmed_case

import seaborn as sns
sns.barplot(x='state',y='confirmed',data=top_confirmed_case,hue='state')
plt.title("Top Covid-19 Affected State",fontsize=20,color='red')
sns.set(rc={'figure.figsize':(10,7)})

max_deaths_case = today.sort_values('deaths',ascending=False)
top_deaths_case= max_confirmed_case.head(5)
top_deaths_case

sns.barplot(x='state',y='deaths',data=top_confirmed_case,hue='state')
plt.title("Top Covid-19 Deaths State",fontsize=20,color='red')
sns.set(rc={'figure.figsize':(10,7)})

max_cured_case = today.sort_values('cured',ascending=False)
top_cured_case= max_confirmed_case.head(5)
top_cured_case

sns.barplot(x='state',y='cured',data=top_confirmed_case,hue='state')
plt.title("Top Covid-19 cured State",fontsize=20,color='red')
sns.set(rc={'figure.figsize':(10,7)})

maharashtra = covid[covid.state == 'Maharashtra']
maharashtra.head()
maharashtra.tail()
maharashtra.shape


sns.lineplot(x='date',y='confirmed',data=maharashtra,hue='state')
sns.set(rc={'figure.figsize':(30,8)})
plt.show()

sns.lineplot(x='date',y='deaths',data=maharashtra,hue='state')
sns.set(rc={'figure.figsize':(30,8)})
plt.show()

sns.lineplot(x='date',y='cured',data=maharashtra,hue='state')
sns.set(rc={'figure.figsize':(30,8)})
plt.show()

covid.head()

X = covid.iloc[:,4:].values
y = covid.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)


print(regressor.coef_)

print(regressor.intercept_)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


df1 = {'Actual deaths':y_test,
'Predicted deaths':y_pred}
df1 = pd.DataFrame(df1,columns=['Actual deaths','Predicted deaths'])
print(df1)

line_chart1 = plt.plot(X_test,y_pred, '--', c ='red')
line_chart2 = plt.plot(X_test,y_test, ':', c='blue')







