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




def ComputelinearK(X1,X2):
	K1 = np.dot(X1, np.transpose(X2))
	return K1


def ComputeGaussianK(X1,X2,spread,n1,n2):
	K2 = np.zeros((n1,n2))
	for i in xrange(n1):
		for j in xrange(n2):
			K2[i,j] = np.exp(-(np.linalg.norm(X1[i]-X2[j])**2/(2*spread)))
	return K2 


def ComputequadraticK(X1,X2):
	K3 = np.dot(X1**2, np.transpose(X2**2))
	return K3


def ComputeKernal(Kernaltype,X1,X2,spread,n1,n2):
	if (Kernaltype == 'linear'):
		K = ComputelinearK(X1,X2)
	elif (Kernaltype =='quadratic'):
		K= ComputequadraticK(X1,X2)
	elif (Kernaltype == 'Gaussian'):
		K = ComputeGaussianK(X1,X2,spread,n1,n2)
	else:
		print ('wrong kernal type!')

	return K




def PredictY(X2,Kernaltype,X,spread,n,n2):
	Y2_pred = np.zeros(n2)
	K = ComputeKernal(Kernaltype,X,X,spread,n,n)
	alpha = np.dot(np.linalg.pinv(K),Y)
	for j in xrange(n2):
		for i in xrange(n):
			K_x_z = ComputeKernal(Kernaltype, X[i],X2[j],spread,1,1)
			Y2_pred[j]+=alpha[i]*K_x_z
	return Y2_pred

'''
def LRestimate(X,X2,Y,Y2):
	part1 = np.linalg.inv(np.dot(np.transpose(X),X))
	part2 = np.dot(np.transpose(X),Y)
	beta = np.dot(part1, part2)
	Y2_pred = np.dot(X2,beta)
	accuracy = sum(np.sign(Y2_pred) == Y2)/float(n2)
	print 'accuracy2 is: '
	print accuracy
	return accuracy
'''

def ComputeAccuracy(X2,d,epsilon,ita,Y,X,Y2,Kernaltype,spread,n,n2):
	Y2_pred = PredictY(X2,Kernaltype,X,spread,n,n2)
	accuracy = sum(np.sign(Y2_pred) == Y2)/float(n2)
	print 'Kernaltype is: '
	print Kernaltype
	print 'epsilon is: '
	print epsilon
	print 'ita is: '
	print ita 
	print 'spread is: '
	print spread 
	print 'accuracy is: '
	print accuracy
	return accuracy


if __name__ == "__main__":

	if (len(sys.argv) == 7):
		filename = sys.argv[1]
		filename2 = sys.argv[2]
		epsilon = float(sys.argv[3])
		ita = float(sys.argv[4])
		Kernaltype = sys.argv[5]
		spread = float(sys.argv[6])
	else:
		filename = sys.argv[1]
		filename2 = sys.argv[2]
		epsilon = float(sys.argv[3])
		ita = float(sys.argv[4])
		Kernaltype = sys.argv[5]
		spread = 0

		print Kernaltype

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


	ComputeAccuracy(X2,d,epsilon,ita,Y,X,Y2,Kernaltype,spread,n,n2)








