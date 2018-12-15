import numpy
import matplotlib.pyplot as pyplot

from sklearn import linear_model
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

x_f1, y_f1 = make_friedman1(n_samples=100, n_features=7, random_state=0)

poly = PolynomialFeatures(degree=2)
x_f1_poly = poly.fit_transform(x_f1)

pyplot.figure()
pyplot.scatter(x_f1[:, 2], y_f1, c="#333333")
pyplot.show()

x_train, x_test, y_train, y_test = train_test_split(x_f1_poly, y_f1, random_state=0)

_lin_reg = LinearRegression().fit(x_train, y_train)

print('\nLinear model coeff (w): \n{0}'.format(_lin_reg.coef_))
print('\n------------------------------------------------------------\n')
print('Linear model intercept (b):{:.3f}'.format(_lin_reg.intercept_))
print('\n------------------------------------------------------------\n')

__lin_reg = Ridge().fit(x_train, y_train)

print('(Poly deg 2 + ridge) Linear model coeff (w) : \n{0}'.format(__lin_reg.coef_))
print('\n------------------------------------------------------------\n')
print('(Poly deg 2 + ridge) Linear model intercept (b):{:.3f}'.format(__lin_reg.intercept_))
print('\n------------------------------------------------------------\n')


print('(Poly deg 2 + ridge) R-squared score (training):{:.3f}'.format(__lin_reg.score(x_train, y_train)))
print('(Poly deg 2 + ridge) R-squared score (test): {:.3f}\n'.format(__lin_reg.score(x_test, y_test)))
