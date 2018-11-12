# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:59:18 2018

@author: Lenovo-g500
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plp

dataframe = pd.read_csv("house.csv")
print(dataframe)

#y = dataframe ['loyer'].values
#x = dataframe ['surface'].values

#plp.scatter(x,y)

plp.plot(dataframe['surface'], dataframe['loyer'],'ro',markersize=4)
plp.show

x = np.matrix([np.ones(dataframe.shape[0]),dataframe['surface'].as_matrix()]).T

y = np.matrix(dataframe['loyer']).T

theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

print(theta)


y = dataframe ['loyer'].values
x = dataframe ['surface'].values

A = np.vstack([x, np.ones(len(x))]).T

print("A equals = ", A)
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, c)

#the equation is Y=mx+c

plp.plot(x, y, 'o', label='Original data', markersize=2)
plp.plot(x, m*x + c, 'r', label='Fitted line')
plp.legend()
plp.show()

print("estimation est ",m*35 + c)

print(m)