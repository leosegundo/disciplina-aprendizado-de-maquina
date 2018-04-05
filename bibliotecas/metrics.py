from stats import std

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
