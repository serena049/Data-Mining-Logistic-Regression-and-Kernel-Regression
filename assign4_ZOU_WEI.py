#!/usr/bin/python 
# -*- coding: utf-8 -*-

#### Using Python 2.7
#### HW4
#### By Wei Zou


import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import random
import sys



def ComputegradientK(yk,xk,w):
	z = -yk*np.dot(w,xk)
	gradientK = (1/(1+np.exp(-z)))*yk*xk 
	return gradientK


def Computew(d,epsilon, ita, Y,X,n):
	w = np.random.rand(d)
	#w_prev = np.ones(d)

	while True:
		w_prev = w.copy()
		list_to_use = range(n)
		random.shuffle (list_to_use)
		for k in list_to_use:
			yk = Y[k]
			xk = X[k]
			gradientK = ComputegradientK(yk,xk,w)
			w = w+ita*gradientK
		if np.sqrt(np.dot((w-w_prev), (w-w_prev)))<=epsilon:
			break
	return w 




def PredictY(X2,w,n2):
	Y2_pred = np.zeros(n2)
	for i in xrange(n2):
		if np.dot(w, X2[i]) >=0.5:
			Y2_pred[i]=1
		else:
			Y2_pred[i]=-1
	return Y2_pred



def ComputeAccuracy(X2,d,epsilon,ita,Y,X,Y2,n,n2):
	w = Computew(d,epsilon,ita,Y,X,n)
	Y2_pred = PredictY(X2,w,n2)
	accuracy = sum(Y2_pred == Y2)/float(n2)
	print 'epsilon is: '
	print epsilon
	print 'ita is: '
	print ita
	print 'accuracy is: '
	print accuracy
	return accuracy


if __name__ == "__main__":

	filename = sys.argv[1]
	filename2 = sys.argv[2]
	epsilon = float(sys.argv[3])
	ita = float(sys.argv[4])


# Training data
	data = np.genfromtxt(filename, delimiter=',', dtype='float')
	d = np.shape(data)[1] # number of columns
	n = np.shape(data)[0] # number of obs
	Y = data[:,-1]
	X10 = data[:,0:-1]
	X = np.ones((n,d))
	X[:,:-1]=X10



# Testing data 
	data2 = np.genfromtxt(filename2, delimiter = ',', dtype = 'float')
	d2 = np.shape(data2)[1]
	n2 = np.shape(data2)[0]
	Y2 = data2[:,-1]
	X20 = data2[:,0:-1]
	X2 = np.ones((n2,d2))
	X2[:,:-1]=X20


	ComputeAccuracy(X2,d,epsilon,ita,Y,X,Y2,n,n2)








