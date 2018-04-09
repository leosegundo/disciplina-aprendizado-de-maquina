from stats import std, diff
import numpy as np

def mse(y_true, y_pred):
	lista = [std((y_true[i] - y_pred[i]),2) for i in range(0,len(y_true))]
	soma = sum(lista)
	return soma/len(y_true)

def rmse(y_true, y_pred):
	mse_ = mse(y_true,y_pred)
	return std(mse_,(1/2))

def mae(y_true, y_pred):
	lista = [abs(y_true[i] - y_pred[i]) for i in range(0,len(y_true))]
	soma = sum(lista)
	return soma/len(y_true)

def accuracy(y_true,y_pred):
	TP = 0
	TN = 0
	for i in range(len(y_pred)):
		if y_true[i]==y_pred[i]==1:
			TP += 1
		if y_true[i]==y_pred[i]==0:
			TN += 1
	return((TP+TN) / len(y_true))


def precision(y_true,y_pred):
	true_positive = [ (diff(y_true[i],y_pred[i])) for i in range(0,len(y_true))]


	return(true_positive.count(1) / len(true_positive))

