import numpy as np
import pandas as pd
from base import Transformer
from GeneticAlgorithm import Evolution

class BaseDtrsm(Transformer):
	def __init__(self, lpn, lnp, lbp, lbn):
		self._X = None
		self._y = None
		self._C = 0
		self._D = 0
		self.LPN = lpn
		self.LNP = lnp
		self.LBP = lbp
		self.LBN = lbn 
		self.alpha = (self.LPN - self.LBN) / (self.LPN - self.LBN + self.LBP)
		self.beta = self.LBN / (self.LBN + self.LNP - self.LBP)
		self.preserved_attrs = None
	
	def partition(self, R=[]):
		X = self._X
		R = list(R) if len(R) else list(self._C)
		X = pd.DataFrame(X)
		eq = X.T.groupby(R).groups.values()
		return eq
		
	def measurement(self, R):
		# preprocess the attribute set
		R = np.array(list(R))
		R = np.arange(R.size)[R == '1']
		if R.size == 0:
			return 0
		# get equivalence class
		ER = self.partition(R)
		return ER, R
	
	def customized_index(self, R):
		pass
	
	def reduction(self):
		# reduct attributes based on the measurement above
		evo = Evolution((20, self._X.shape[0]), 500, self.customized_index)
		best_code = evo.launch()
		R = np.arange(1, X.shape[0] + 1) * best_code
		R = R[R > 0] - 1
		self.preserved_attrs = R
	
	def _fit(self, X, y):
		self._X = X
		self._y = y
		self._C = np.arange(X.shape[0])
		self._D = np.arange(y.shape[0])
		self.reduction()
		return X[self.preserved_attrs,:].T
	

class MinCostReduct(BaseDtrsm):
	'''
	以决策损失最小化来约简属性
	参数：
	[lpn] 将假例判断成真例的损失
	[lnp] 将真例判断成假例的损失
	[lbp] 将真例判断为等待进一步观察的损失
	[lbn] 将假例判断成等待进一步观察的损失
	调用方法：
	>>> mcr = MinCostReduct(6, 3, 1, 3)
	>>> # X 以列向量输入，m个nx维样本的样本集 X 输入为 X.shape 为 (nx,m)，y.shape 为 (ny,m)
	>>> X = mcr.fit_transform(X, y)
	>>> # 可通过重写 MinCostReduct 类的 customized_index 方法来自定义 fitness 度量
	'''
	
	def __init__(self, lpn, lnp, lbp, lbn):
		super().__init__(lpn, lnp, lbp, lbn)
	
	def measurement(self, R):
		'''Cost-preserved : the cost of making all decisions of three types'''
		ER, R = super().measurement(R)
		loss = 0
		for eqcls in ER:
			# get the weighest dmax class
			weighest = max(Counter(y[0, eqcls]).values()) / y[0, eqcls].size
			if weighest >= self.alpha:
				loss += (1 - weighest) * self.LPN * len(eqcls)
			elif weighest > self.beta:
				loss += (1 - weighest) * self.LBN * len(eqcls)
				loss += weighest * self.LBP * len(eqcls)
			else:
				loss += weighest * self.LNP * len(eqcls)
		# 根据约简集的长度和损失来计算 fitness
		return loss
	
	def customized_index(self, R):
		loss = self.measurement(R)
		return 1 / (loss + (len(R.strip("0")) / self._X.shape[0]) ** 0.3)
	
		
class RegionPreservedReduct(BaseDtrsm):
	'''
	保持决策类的非负域不变以约简属性
	参数：
	[lpn] 将假例判断成真例的损失
	[lnp] 将真例判断成假例的损失
	[lbp] 将真例判断为等待进一步观察的损失
	[lbn] 将假例判断成等待进一步观察的损失
	调用方法：
	>>> rpr = RegionPreservedReduct(6, 3, 1, 3)
	>>> # X 以列向量输入，m个nx维样本的样本集 X 输入为 X.shape 为 (nx,m)，y.shape 为 (ny,m)
	>>> X = rpr.fit_transform(X, y)
	>>> # 可通过重写 RegionPreservedReduct 类的 customized_index 方法来自定义 fitness 度量
	'''

	def __init__(self, lpn, lnp, lbp, lbn):
		super().__init__(lpn, lnp, lbp, lbn)
		
	def measurement(self, R):
		# non-negative region preserved
		ER, R = super().measurement(R)
		nneg_size = 0	# 非负域的样本个数
		for eqcls in ER:
			weighest = max(Counter(y[0, eqcls]).values()) / y[0, eqcls].size
			if weighest >= self.beta:
				nneg_size += 1
		return nneg_size
	
	def customized_index(self, R):
		nneg_size = self.measurement(R)
		return nneg_size / self._C.size + (self._C.size / len(R.strip("0"))) ** 0.3
	
	
class EntropyPreservedReduct(BaseDtrsm):
	'''
	保持系统的条件熵不变以约简属性
	参数：
	[lpn] 将假例判断成真例的损失
	[lnp] 将真例判断成假例的损失
	[lbp] 将真例判断为等待进一步观察的损失
	[lbn] 将假例判断成等待进一步观察的损失
	调用方法：
	>>> epr = EntropyPreservedReduct(6, 3, 1, 3)
	>>> # X 以列向量输入，m个nx维样本的样本集 X 输入为 X.shape 为 (nx,m)，y.shape 为 (ny,m)
	>>> X = epr.fit_transform(X, y)
	>>> # 可通过重写 EntropyPreservedReduct 类的 customized_index 方法来自定义 fitness 度量
	'''
	
	def __init__(self, lpn, lnp, lbp, lbn):
		super().__init__(lpn, lnp, lbp, lbn)
	
	def measurement(self, R):
		# conditional entropy
		ER, R = super().measurement(R)
		conditional_entropy = 0
		for eq in ER:
			pXi_y_distr = pd.Series(self._y[0, eq]).value_counts().to_numpy()
			pi = pXi_y_distr / pXi_y_distr.sum()
			conditional_entropy -= eq.size / self._X.shape[1] * (pi * np.log(pi)).sum()
		return conditional_entropy
		
	def customized_index(self, R):
		entropy = self.measurement(R)
		return np.exp(-entropy) + (self._C.size / len(R.strip("0"))) ** 0.3