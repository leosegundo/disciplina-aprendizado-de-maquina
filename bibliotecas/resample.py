import numpy as np

def split_train_test(n_elem, perc_train, seed):
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

# def split_k_fold2(n_elem, n_splits = 3, shuffle = True, seed = 0):
# 	a = [ i for i in range(n_elem) ]
# 	np.random.seed(seed)
# 	if shuffle:
# 		np.random.shuffle(a)
    
# 	n_test = int(round(n_elem * (1.0 / n_splits)))
# 	n_train = n_elem - n_test
    
# 	train = []
# 	test = []
# #     print(a)
# #     print(n_test)
    
# 	for i in range(n_splits):
# #         print(i)
# 		flag = False
# 		flag_ = False
        
# 		if(i == n_splits-1):
# 			flag_ = True
		
# 		train_ = []
# 		test_ = []
        
# 		if(flag_):
# 			for j in range(n_elem):
# 				if(j <= i*n_test):
# 					train_.append(a[j])
# 				else:
# 					test_.append(a[j])
# 		else:
# 			for j in range(n_elem):
# 				if( j >= (i*n_test) and j < ( (i+1)*n_test )):
# 					flag = True
# 				else:
# 					flag = False
# 				if(flag):
# 					print("não entrous")
# 					test_.append(a[j])
# 				else:
# 					print("entrou")
# 					print(str(j) + " " + str((i+1)*n_test) )
# 					print(i)
# 					train_.append(a[j])
            
# 		train.append(train_)
# 		test.append(test_)
            
# 	return np.array(train),np.array(test)



