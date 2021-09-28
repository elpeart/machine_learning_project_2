# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:28:06 2020

@author: elpea
"""

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
# load the training dataset
train_data = loadtxt('zipcode_train.csv', delimiter=',')
# split into input (X) and output (y) variables
X = train_data[:,0:16]
y = train_data[:,16]
y = to_categorical(y)
# load the testing dataset
test_data = loadtxt('zipcode_test.csv', delimiter=',')
# split into input (X) and output (y) variables
X1 = test_data[:,:16]
y1 = test_data[:,16]
y1 = to_categorical(y1)

for n in [5, 10, 13]:
# define the keras model
    model = Sequential()
    model.add(Dense(n, input_dim=16, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=15, batch_size=10)
    
    # evaluate the model
    _, train_acc = model.evaluate(X, y)
    _, test_acc = model.evaluate(X1, y1)

    print('With n = %d, Training accuracy: %.3f, Testing accuracy: %.3f' % (n, train_acc, test_acc))