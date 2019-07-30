import numpy as np
import pandas as pd
from pprint import pprint
from collections import Counter
from base import Transformer
from GeneticAlgorithm import Evolution

df = pd.read_csv("/Users/apple/Data Sets/Mushroom Data Set/agaricus-lepiota.data", header=None)
choose_index = np.random.randint(8000,size=1000)
df = df.iloc[choose_index, :]
X = df.iloc[:,1:].to_numpy().T
y = np.array([df.iloc[:,0].to_numpy()]).reshape(1,-1)


class BaseDtrsm(Transformer):
	def __init__(self, lpn, lnp, lbp, lbn):
		self._X = None
		self._y = None
		self._C = 0
		self._D = 0
		self.LPN = lpn # 6
		self.LNP = lnp # 3
		self.LBP = lbp # 1
		self.LBN = lbn # 3
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
		ER = self.partition(R)
		return ER, R
	
	def reduction(self):
		# reduct attributes based on the measurement above
		evo = Evolution((20, self._X.shape[0]), 500, self.measurement)
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
		return 1 / (loss + (R.size / self._X.shape[0]) ** 0.3)
	

		
class RegionPreservedReduct(BaseDtrsm):
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
		return nneg_size / self._C.size + (self._C.size / R.size) ** 0.3
		

class EntropyPreservedReduct(BaseDtrsm):
	def __init__(self, lpn, lnp, lbp, lbn):
		super().__init__(lpn, lnp, lbp, lbn)
	
	def measurement(self, R):
		# conditional entropy
		ER, R = super().measurement(R)
		for eq in ER:
			pXi = eq.size / self._X.shape[1]
			partition = list(pd.DataFrame(y[:, eq]).T.groupby([0]).groups.values())
			partition = np.array(partition)
#			getlen = np.vectorize(lambda x : x.size)
#			partition = getlen(partition)
			print(partition)
			print(len(eq))
			break
		



if __name__ == "__main__":
	epr = EntropyPreservedReduct(6, 3, 1, 3)
	X = epr.fit_transform(X, y)