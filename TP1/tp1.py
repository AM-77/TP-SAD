import pandas 
import numpy
from matplotlib import pyplot 

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataframe = pandas.read_csv("house.csv")

"""
print(dataframe)

y = dataframe ['loyer'].values
x = dataframe ['surface'].values
pyplot.scatter(x,y)
pyplot.show()

"""

"""
pyplot.plot(dataframe['surface'], dataframe['loyer'], 'ro',markersize=2)
pyplot.show()

"""

x = numpy.matrix([numpy.ones(dataframe.shape[0]),dataframe['surface'].as_matrix()]).T
y = numpy.matrix(dataframe['loyer']).T

theta = numpy.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print(theta)

pyplot.xlabel("Surface")
pyplot.ylabel("Loyer")
pyplot.plot(dataframe["surface"], dataframe["loyer"], 'ro', markersize=2, c="#333333")
pyplot.plot([0, 250], [theta.item(0), theta.item(0) + 250 * theta.item(1)], c="#119955")

pyplot.show()

print("Estimation : ", theta.item(1) * 35 + theta.item(0))

"""
print("----------------------------------------------------------")
print(regr.predict([[35]]))
print("----------------------------------------------------------")

"""
print("\n\n\n\n")

regr = linear_model.LinearRegression()
lng = regr.fit(x, y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y)
l = regr.fit(X_train, Y_train)

print(l.coef_)
print(l.intercept_)
print("\n\n")
print(l.score(X_train, Y_train))
print(l.score(X_test, Y_test))

