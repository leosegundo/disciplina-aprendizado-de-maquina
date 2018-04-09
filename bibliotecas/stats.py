import numpy as np
from sklearn import	metrics	

def std(x,y):
	return x ** y

def mean(x):
	return (sum(x)/len(x))

def var(x):
	media = mean(x)
	soma = 0
	for valor in x:
		soma += std((valor-media),2)
	variancia = soma /(len(x))
	return variancia

def stdev(x):
	variancia = var(x)
	return std(variancia,(1/2))

def diff(x,y):
	if np.array_equal(x,y):
		return 1
	else:
		return 0;

	
def confusion_matrix(y_true,y_pred):
	return metrics.confusion_matrix(y_true,	y_pred)	
