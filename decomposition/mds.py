import numpy as np
from base import Transformer

class MDS(Transformer):
	def __init__(self, n_components = 2):
		self.n_components = n_components
	
	def _fit(self, X, n_components):
		# 计算距离平方矩阵 Dmxm
		X = X.T
		d, m = X.shape # d 为训练集的属性维度，m 为其样本数量
		D = np.zeros((m, m))
		for i in range(m):
			for j in range(i+1):
				D[i,j] = np.square(X[:,i] - X[:,j]).sum()
				D[j,i] = D[i,j]
		# 计算 B = ZtZ
		B = np.zeros((m, m))
		for i in range(m):
			for j in range(i+1):
				B[i,j] = (D[i,:].sum() + D[:,j].sum() - D.sum()/m)/(2*m) - D[i,j]/2
				B[j,i] = B[i,j]
		# B 特征分解
		eigval, eigvec = np.linalg.eig(B)
		# 属性约简
		choose_index = np.argsort(eigval)[-1:-1-n_components:-1]
		eigval = np.real(eigval[choose_index])
		eigvec = np.linalg.inv(eigvec)
		eigvec = np.real(eigvec[choose_index])
		# 计算 Z
		Z = np.diag(np.sqrt(eigval)).dot(eigvec)
		return Z.T
		
