import stats as st

def standardize(X):
	mean = st.mean(X)

	x = [st.std((X[i] - mean),2) for i in range(0,len(X))]
	
	value_sd = (sum(x) / len(X))
	standard_deviation = st.std(value_sd,(1/2))
	print(standard_deviation)

	y =  [((X[i] - mean)/standard_deviation) for i in range(0,len(X))]
	
	return y


def normalize(X):
	y = [((X[i] - min(X))/(max(X) - min(X))) for i in range(0,len(X))] 
	return y



#a = [1,2,3,4,5]

#k = standardize(a)
#print(k)

#k = normalize(a)
#print(k)
