import dist_metrics as dm
import numpy as np

class KNNClassifier:
    k = 0
    metric = None
    p = None
    X = None
    y = None
    
    def __init__(self, k=5, metric='minkowski', p=2):
        self.k = k
        self.metric = metric
        self.p = p

    def get_neighbors(self,X_train, test_row,k):
        if(self.metric == 'minkowski'):
            distances = dm.minkowski_distance(X_train,test_row, self.p)
        elif(self.metric == 'euclidian'):
            distances = dm.euclidean_distance(X_train,test_row)
        elif(self.metric == 'euclidian'):
            distances = dm.manhattan_distance(X_train,test_row)
        elif(self.metric == 'chebyshev'):
            distances = dm.chebyshev_distance(X_train,test_row)
        idx_sort = np.argsort(distances)
        return idx_sort[1:k+1]

    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self,X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            idx_sort = self.get_neighbors(self.X, X_test[i],self.k)
            output_values = self.y[idx_sort]
            counts = np.unique(output_values, return_counts=True)
            idx_max = np.argmax(counts[1])
            y_pred.append(counts[0][idx_max])
        return y_pred


class KNNRegressor:
    k = 0
    metric = None
    p = None
    X = None
    y = None
    def __init__(self, k=5, metric='minkowski', p=2):
        self.k = k
        self.metric = metric
        self.p = p

    def get_neighbors(self,X_train, test_row,k):
        if(self.metric == 'minkowski'):
            distances = dm.minkowski_distance(X_train,test_row, self.p)
        elif(self.metric == 'euclidian'):
            distances = dm.euclidean_distance(X_train,test_row)
        elif(self.metric == 'euclidian'):
            distances = dm.manhattan_distance(X_train,test_row)
        elif(self.metric == 'chebyshev'):
            distances = dm.chebyshev_distance(X_train,test_row)
        idx_sort = np.argsort(distances)
        return idx_sort[1:k+1]

    def predict(self, X_test):
        y_pred = []
        for i in range(X_teste.shape[0]):
            idx_sort = get_neighbors(self.X, X_test[i], self.k)
            output_values = self.y[idx_sort]
            y_pred.append(np.sum(output_values) / output_values.shape[0])
        return y_pred