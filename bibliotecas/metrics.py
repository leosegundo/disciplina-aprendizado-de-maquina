import stats as st
import numpy as np

def mse(y_true, y_pred):
	lista = [st.std((y_true[i] - y_pred[i]),2) for i in range(0,len(y_true))]
	soma = sum(lista)
	return soma/len(y_true)

def rmse(y_true, y_pred):
	mse_ = mse(y_true,y_pred)
	return st.std(mse_,(1/2))

def mae(y_true, y_pred):
	lista = [abs(y_true[i] - y_pred[i]) for i in range(0,len(y_true))]
	soma = sum(lista)
	return soma/len(y_true)

def accuracy(y_true,y_pred):
	matriz = st.confusion_matrix(y_true,y_pred)
	return  np.sum(np.diagonal(matriz))/np.sum(matriz)


def precision(y_true, y_pred):
	matriz = st.confusion_matrix(y_true,y_pred)
	values = []
	for i in range(0,len(matriz)):
		TP = matriz[i][i]
		FP = np.sum(matriz[i,:])
		values.append((TP/(TP+FP)))

	return values

def recall(y_true,y_pred):
    matriz = st.confusion_matrix(y_true,y_pred)
    values = []
    for i in range(0,len(matriz)):
    	TP = matriz[i][i]
    	FN = np.sum(matriz[:,i])
    	values.append((TP/(TP+FN	)))

    return values

def f1_measure(y_true,y_pred):
	prec = np.array(precision(y_true,y_pred))
	rec = np.array(recall(y_true,y_pred))
	a = 2 * prec
	b = a * rec
	return (b) / (prec + rec)