import numpy as np
def p_root(value, root):
     
    root_value = 1 / float(root)
    return np.round(value ** root_value, 3)
 
def minkowski_distance(X, row, p):
	# pass the p_root function to calculate
    # all the value of vector parallely 
    return (p_root(sum(pow(abs(a-b), p)
            for a, b in zip(X, row)), p))

def euclidean_distance(X, row):
	return minkowski_distance(X,row,2)

def manhattan_distance(X, row):
	return minkowski_distance(X,row,1)


def chebyshev_distance(X, row):
	d = [abs(a-b) for a,b in zip(X,row)]
	return max(d)