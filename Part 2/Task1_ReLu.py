# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:30:28 2020

@author: elpea
"""

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
# load the training dataset
dataset = loadtxt('regression.tra.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8:15]
# load the testing dataset
dataset = loadtxt('regression.tst.csv', delimiter=',')
# split into input (X) and output (y) variables
X1 = dataset[:,:8]
y1 = dataset[:,8:]
for n in [1, 4, 8]:
    # define the keras model
    model = Sequential()
    model.add(Dense(n, input_dim=8, activation='relu'))
    model.add(Dense(7, activation='linear'))
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=15, batch_size=10)

    ypred = model.predict(X)
    y1pred = model.predict(X1)

    print("y1 train MSE with n = %d: %.4f" % (n, mean_squared_error(y[:,0], ypred[:,0])))
    print("y2 train MSE with n = %d: %.4f" % (n, mean_squared_error(y[:,1], ypred[:,1])))
    print("y3 train MSE with n = %d: %.4f" % (n, mean_squared_error(y[:,2], ypred[:,2])))
    print("y4 train MSE with n = %d: %.4f" % (n, mean_squared_error(y[:,3], ypred[:,3])))
    print("y5 train MSE with n = %d: %.4f" % (n, mean_squared_error(y[:,4], ypred[:,4])))
    print("y6 train MSE with n = %d: %.4f" % (n, mean_squared_error(y[:,5], ypred[:,5])))
    print("y7 train MSE with n = %d: %.6f \n" % (n, mean_squared_error(y[:,6], ypred[:,6])))
    print("y1 test MSE with n = %d: %.4f" % (n, mean_squared_error(y1[:,0], y1pred[:,0])))
    print("y2 test MSE with n = %d: %.4f" % (n, mean_squared_error(y1[:,1], y1pred[:,1])))
    print("y3 test MSE with n = %d: %.4f" % (n, mean_squared_error(y1[:,2], y1pred[:,2])))
    print("y4 test MSE with n = %d: %.4f" % (n, mean_squared_error(y1[:,3], y1pred[:,3])))
    print("y5 test MSE with n = %d: %.4f" % (n, mean_squared_error(y1[:,4], y1pred[:,4])))
    print("y6 test MSE with n = %d: %.4f" % (n, mean_squared_error(y1[:,5], y1pred[:,5])))
    print("y7 test MSE with n = %d: %.6f" % (n, mean_squared_error(y1[:,6], y1pred[:,6])))