import numpy as np



def split_stratified_train_test(y, perc_train, seed):
	indices = [ i for i in range(0,len(y))]
	np.random.seed(seed)
	np.random.shuffle(indices)

	n = int(len(indices)*perc_train)
	
	idx_train = [ indices[i] for i in range(0, n)]
	idx_test = [indices[i] for i in range(n, len(indices))]

	return idx_train,idx_test

	

def split_train_test(n_elem, perc_train, seed	):
	indices = [ i for i in range(n_elem)]
	np.random.seed(seed)
	np.random.shuffle(indices)

	n = int(len(indices)*perc_train)
	
	idx_train = [ indices[i] for i in range(0, n)]
	idx_test = [indices[i] for i in range(n, len(indices))]

	return idx_train,idx_test

def split_k_fold(n_elem, n_splits=3, shuffle=True, seed=0):
	indices = [ i for i in range(n_elem)]
	np.random.seed(seed)
	if shuffle:
		np.random.shuffle(indices)

	mean_size = round(n_elem/n_splits)


	train = []
	test = []
	
	for i in range(n_splits):
		count = 0
		flag = False
		
		train1 = []
		test1 = []

		if(i == n_splits-1):
			flag =True
		
		for j in range(n_elem):
			if( j >= (i*mean_size) and j < ( (i+1)*mean_size )):
				test1.append(indices[j])
			else:
				train1.append(indices[j])

		train.append(train1)
		test.append(test1)

	return np.array(train),np.array(test)
