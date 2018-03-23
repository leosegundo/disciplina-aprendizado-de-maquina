import numpy as np

def split_train_test(n_elem, perc_train, seed):
	indices = [ i for i in range(n_elem)]
	np.random.seed(seed)
	np.random.shuffle(indices)

	n = int(len(indices)*perc_train)
	
	idx_train = [ indices[i] for i in range(0, n)]
	idx_test = [indices[i] for i in range(n, len(indices))]

	return idx_train,idx_test
