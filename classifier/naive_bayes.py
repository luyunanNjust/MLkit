import numpy as np
from functools import reduce
from BaseClassifier import BaseClassifier


class DiscreteBayes(BaseClassifier):
	"""
	【适用于属性为离散值的贝叶斯分类器】
	"""
	def __init__(self):
		super().__init__()
		self._X_test = None
		self._y_test = None
		self._C = None

	# 训练学习器
	def fit(self, X, y):
		self._C = sorted(list(np.unique(y)))
		super().fit(X, y)
	
	# 预测重用代码
	def _pred(self, i):
		x = self._X_test[i]
		C_distribution = dict().fromkeys(self._C)
		# 计算公式 P(ci|x) = P(x|ci) * P(ci) / P(x)
		for ci in self._C:
			Xci = self._X[self._y == ci, :]  # 剥出有用数据集
			Pci = (self._y[self._y == ci]).size / self._y.size # P(ci)
			# 计算 P(x|ci)
			P_x_ci = 1
			multi_prod = np.vectorize(
				lambda di : (Xci[:, di][Xci[:, di] == x[di]]).size / Xci[:, di].size
			)
			pd_array = multi_prod(range(Xci.shape[1]))
			C_distribution[ci] = reduce(lambda x,y:x*y, pd_array)
		return C_distribution
	
	# 获得决策类别的概率分布
	def predict_proba(self, X):
		self._X_test = X
		get_distribution = np.vectorize(self._pred)
		distri_arr = get_distribution(range(X.shape[0]))
		
		# 将概率分布的字典形式转化成数组形式
		res = list()
	
		def trans(distr):
			C = [0] * len(self._C)
			for it in distr.items():
				C[it[0] - self._C[0]] = it[1]
			res.append(C)
			return 0
		
		trans = np.vectorize(trans) # 向量化
		trans(distri_arr)
		return np.array(res[1:])
	
	# 预测
	def predict(self, X):
		self._X_test = X
		get_distribution = np.vectorize(self._pred)
		distri_arr = get_distribution(range(X.shape[0]))
		# 选取概率分布中最大的（向量化）
		get_max = lambda distr : max(distr.items(), key = lambda x : x[1])[0]
		get_max = np.vectorize(get_max)
		return np.array(get_max(distri_arr))

