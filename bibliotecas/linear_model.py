import stats as st
import numpy as np

class SimpleLinearRegression:

	b0 = b1 = None
	def __init__(self):
		self.b0 = 0.0
		self.b1 = 0.0

	def fit(self, X, y):
		media_x = st.mean(X)
		media_y = st.mean(y)

		x_less_lineX = [(X[i] - media_x) for i in range(0,len(X))]
		y_less_liney = [(y[i] - media_y) for i in range(0,len(y))]

		x_less_lineX_std = [st.std(valor,2) for valor in x_less_lineX]

		x_mult_y = [(x_less_lineX[i] * y_less_liney[i]) for i in range(0,len(x_less_lineX))]

		self.b1 = sum(x_mult_y)/sum(x_less_lineX_std)

		self.b0 = media_y - (self.b1 * media_x)

		return self

	def predict(self, X):
		pred = [((self.b1 * X[i]) + self.b0 ) for i in range(0,len(X))]
		return pred

class LogisticRegression:
	
	
	np.random.seed(3)
	num_pos = 5000
	epochs = 0
	learning_rate = 0
	
	def __init__(self, epochs, learning_rate):
		self.epochs = epochs
		self.learning_rate = learning_rate
		
	# Bivariate normal distribution mean [0, 0] [0.5, 4], with a covariance matrix
	#subset1 = np.random.multivariate_normal([0, 0], [[1, 0.6],[0.6, 1]], num_pos)
	#subset2 = np.random.multivariate_normal([0.5, 4], [[1, 0.6],[0.6, 1]], num_pos)

	#dataset = np.vstack((subset1, subset2))
	#x = np.hstack((np.ones(num_pos*2).reshape(num_pos*2, 1), dataset)) # add 1 for beta_0 intercept
	#label = np.hstack((np.zeros(num_pos), np.ones(num_pos)))
	#y = label.reshape(num_pos*2, 1) # reshape y to make 2D shape (n, 1)
	
	def fit(self,X,y): 
		beta = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
		for step in np.arange(epochs):
		    x_beta = np.dot(X, beta)
		    y_hat = 1 / (1 + np.exp(-x_beta))
		    likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
		    preds = np.round( y_hat )
		    accuracy = np.sum(preds == y)*1.00/len(preds)
		    gradient = np.dot(np.transpose(X), y - y_hat)
		    beta = beta + learning_rate*gradient
		    if( step % 5000 == 0):
		    	print("After step {}, likelihood: {}; accuracy: {}".format(step+1, likelihood, accuracy))
		self.beta = beta
	
	def predict(self, X):
		b0 = self.beta[0]
		x1 = X[:, 0]
		x2 = X[:, 1]

		rt = 1.0 + np.exp(-(b0+(beta[1]*x1)+(beta[2]*x2)))
		return 1.0/rt	