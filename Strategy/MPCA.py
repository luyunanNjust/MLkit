import numpy as np

class MPCA:
	"""
	「参数」
		base_estimator : 基本学习器
		L : 方阵, shape = [决策类别个数, 决策类别个数]
			L[i][j] 表示将第 j 决策类别判成第 i 决策类别的损失
		thetaBP : 数组, shape = [决策类别个数,]
			thetaBP[i] 表示对第 i 类决策类别作出延迟决策相对于作出误判的损失比率
		thetaBN : 同 thetaBP
		**kwargs : base_estimator 的参数
	"""
	def __init__(self, base_estimator, thetaBP, thetaBN, L, **kwargs):
		self.base_estimator = base_estimator
		self._be_params = kwargs
		self.thetaBP = thetaBP
		self.thetaBN = thetaBN
		self.L = L
		self._C = None
		self._X = None
		self._y = None

	def fit(self, X, y):
		self._X = X
		self._y = y
		self._C = np.unique(y)
		return self
	
	def predict(self, X):
		X_train, y_train, X_test = self._X, self._y, X
		y_pred = np.zeros(y_test.size)
		TEST = np.arange(X_test.shape[0])
		
		while True:
			# 新建基本学习器
			estimator = self.base_estimator(self._be_params) if len(self._be_params) else self.base_estimator()
			estimator = estimator.fit(X_train, y_train)
			proba_distr = estimator.predict_proba(X_test)
			
			# 求LPN[i]， 表示将不是第 i 类的样本判为第 i 类的损失
			LPN = np.array([(self.L[ci,:] * proba_distr).mean(axis=0).sum() for ci in self._C])
			LNP = np.array([(self.L[:,ci] * proba_distr).mean(axis=0).sum() for ci in self._C])
			LBP = thetaBP * LNP
			LBN = thetaBN * LPN
			# alpha[i] 表示第 i 类的阈值
			alpha = (LPN - LBN) / (LPN - LBN + LBP)
			beta = LBN / (LBN + LNP - LBP)

			# 求正域与非正域的样本序号
			NPOS = np.array([ti for ti in TEST if not (proba_distr[ti] > alpha).any()])
			POS = np.setdiff1d(TEST, NPOS)
			y_pred[POS] = np.argmax(proba_distr[POS], axis=1)
			
			# 若非正域收敛
			if POS.size == 0:
				break
			
			# 更新训练集
			X_add = X_test[POS,:].reshape(1,-1) if POS.size == 1 else X_test[POS,:]
			X_train = np.concatenate((X_train, X_add))
			y_train = np.concatenate((y_train, y_pred[POS]))
			
			# 缩减测试集
			TEST = NPOS
			
		# 最近邻决策顽固 NPOS
		for i in TEST:
			xi = X_test[i].reshape(1,-1)
			dist = np.power(X_train - xi, 2).sum(axis=1)
			y_pred[i] = y_train[np.argmin(dist)]
		
		return y_pred