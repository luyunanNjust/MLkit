class BaseClassifier:
	"三个方法需重构，其一 fit，其二 predict，其三 score"
	def __init__(self):
		self._X = None
		self._y = None
	
	def fit(self, X, y, sample_weight=None):
		self._X = X
		self._y = y
		return self
	
	def predict(self, X):
		pass
	
	def score(self, X, y):
		pass