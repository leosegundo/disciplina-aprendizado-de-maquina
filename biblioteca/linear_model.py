import stats as st

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

