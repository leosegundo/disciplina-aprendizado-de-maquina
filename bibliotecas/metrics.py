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
	true_positive = [ (y_true[i] == y_pred[i]) for i in range(0,len(y_true))]
	return np.sum(true_positive)/len(y_true)

def precision_score(y_true, y_pred):
    c_tam,c_idx = np.unique(y_true,return_index=True)
    PV = 0
    PF = 0
    for i in range(len(c_tam)):
        for x in range(len(y_true)):
            if (y_pred[x] == y_true[c_idx[i]]) and (y_true[x] == y_pred[x]):
                PV = PV + 1
            if (y_pred[x] == y_true[c_idx[i]]) and (y_true[x] != y_pred[x]):
                PF = PF +1
    PF = PF / len(c_tam)
    return ((PV) / (PV + PF))

def recall(y_true,y_pred):
    c_tam,c_idx = np.unique(y_true,return_index=True)
    PV = 0
    P = 0
    total = 0
    for i in range(len(c_tam)):
        for x in range(len(y_true)):
            if (y_pred[x] == y_true[c_idx[i]]) and (y_true[x] == y_pred[x]):
                PV = PV + 1
            P = P + 1
        total = total + (PV / P)
    return  total

def f1_measure(y_true,y_pred):
    return (2 * precision_score(y_true,y_pred) * recall(y_true,y_pred)) / ( precision_score(y_true,y_pred) + recall(y_true,y_pred)) 