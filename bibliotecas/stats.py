import numpy as np
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
	
