import numpy as np
from functools import reduce
from BaseClassifier import BaseClassifier

class DiscreteNB(BaseClassifier):
	"""
	【适用于属性为离散值的贝叶斯分类器】
	确保决策类的标记值为 0 1 2 3...
	"""
	def __init__(self):
		super().__init__()
		self._X_test = None
		self._y_test = None
		self._C = None
		# 外部可访问属性
		self.class_prior_ = list()

	# 训练学习器
	def fit(self, X, y):
		super().fit(X, y)
		self._C = sorted(list(np.unique(y)))
		for ci in self._C:
			Pci = (self._y[self._y == ci]).size / self._y.size
			self.class_prior_.append(Pci)
		self.class_prior_ = np.array(self.class_prior_)
		return self
	
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
				C[it[0]] = it[1]
			res.append(C)
			return 0
		
		trans = np.vectorize(trans) # 向量化
		trans(distri_arr)
		res = np.array(res[1:])
		res = res / res.sum(axis = 1).reshape(-1,1)
		return res
	
	# 预测
	def predict(self, X):
		self._X_test = X
		get_distribution = np.vectorize(self._pred)
		distri_arr = get_distribution(range(X.shape[0]))
		# 选取概率分布中最大的（向量化）
		get_max = lambda distr : max(distr.items(), key = lambda x : x[1])[0]
		get_max = np.vectorize(get_max)
		return np.array(get_max(distri_arr))

class GaussianNB(DiscreteNB):
	"""
	【适用于属性值连续的贝叶斯分类器】
	确保决策类的标记值为 0 1 2 3...
	"""
	def __init__(self):
		super().__init__()
		self.gauss_func_dict = dict()
	
	def __gauss_distr(self, sigma, miu, x):
		return np.exp(-np.power(x - miu, 2) / (2*sigma*sigma)) / (sigma * np.sqrt(2*np.pi))
	
	def fit(self, X, y):
		self._C = sorted(list(np.unique(y)))
		self._X = X
		self._y = y
		for ci in self._C:
			Xci = X[y == ci]
			gs_func = np.vectorize(lambda i : (Xci[:,i].std(), Xci[:,i].mean()))
			self.gauss_func_dict[ci] = gs_func(range(Xci.shape[1]))
			Pci = (self._y[self._y == ci]).size / self._y.size
			self.class_prior_.append(Pci)
		self.class_prior_ = np.array(self.class_prior_)
		return self
		
	def _pred(self, i):
		x = self._X_test[i]
		C_distribution = dict().fromkeys(self._C)
		# 计算公式 P(ci|x) = IIP(xd|ci) * P(ci) / P(x)
		for ci in self._C:
			p_fea = self.__gauss_distr(self.gauss_func_dict[ci][0], self.gauss_func_dict[ci][1], x)
			C_distribution[ci] = reduce(lambda x,y:x*y, p_fea) * self.class_prior_[ci]
		return C_distribution