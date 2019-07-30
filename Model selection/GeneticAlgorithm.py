import numpy as np
import numpy.random as nr
import pandas as pd
from functools import reduce
from bisect import insort


class Evolution:
	def __init__(self, population_size, epoches, fitness):
		self.POPULATION_SIZE = population_size # (x, y) 种群包含 x 个个体，每个个体由 y 个染色体代表
		self.epoches = epoches
		self.population = nr.randint(2, size=self.POPULATION_SIZE)
		self.fitness = fitness # 确保得分是正数
		
	def initialize_population(self):
		'''初始化 population，通常采用二进制模式'''
		pass
	
	def encode_indv(self, indv_ndarr):
		'''用户自定义，将染色体信息编码成个体信息'''
		return reduce(lambda x, y : str(x) + str(y), indv_ndarr)
		
	def decode_chromo(self, chromosome):
		'''用户自定义，将个体信息编码成染色体字符串信息'''
		return [int(x) for x in chromosome]

	def __roulette_selection(self, chromo_score):
		scores = np.array([x[1] for x in chromo_score])
		posibility = scores / scores.sum()
		indv = [x[0] for x in chromo_score]
		return nr.choice(indv, size=2, p=posibility)

	def __crossover(self, parents_chromos):
		daddy, mummy = parents_chromos[0], parents_chromos[1]
		cut = nr.randint(1, self.POPULATION_SIZE[1] - 1) # cutting point
		# connect the pre-portion of daddy with the post-portion of mummy
		baby = daddy[:cut] + mummy[cut:]
		return baby
		
	def __mutate(self, baby_chromo):
		baby = self.decode_chromo(baby_chromo)
		mut_ind = nr.randint(self.POPULATION_SIZE[1])
		baby[mut_ind] = abs(baby[mut_ind] - 1) # flip baby[mut_ind]
		baby = self.encode_indv(baby)
		return baby


	# 主要代码逻辑
	def launch(self, **kwargs):
		chromo_score = list() # 用以存储种群中每个个体的适应度得分
		individuals = [self.encode_indv(x) for x in list(self.population)]
		# #########################
		# 初始计算种群中每个个体的适应度得分
		for chromo in individuals:
			print(chromo)
			score = self.fitness(chromo, **kwargs)
			chromo_score.append((chromo, score))
		
		# #########################
		# 模拟种群进化过程的主循环
		epoch = 0
		converge_factor = 0 # 为了避免天才父母生出脑残儿童的小概率事件
		while epoch < self.epoches:
			parents = self.__roulette_selection(chromo_score) # 使用轮盘随机选择出生产下一代的父母
			baby = self.__crossover(parents) # 父母 ooxx 基因配对
			baby = self.__mutate(baby) # 随机基因变异
			baby_score = self.fitness(baby, **kwargs) # 计算新个体的适应度得分
			
			# 判断种群适应度是否已经收敛
			worst_indv, min_score = min(chromo_score, key=lambda x : x[1])
			if baby_score <= min_score:
				converge_factor += 0.1
				if converge_factor >= 1:
					break # 如果收敛因子超过 1 就断定种群已经收敛了
			else:  # 如果种群还未收敛，即新一代的适应度得分比上一代最差得分高
				# 优胜劣汰，淘汰最差者，加入新一代
				chromo_score.append((baby, baby_score))
				chromo_score.remove((worst_indv, min_score))
			epoch += 1
			print("generation : %d, baby : %s, and its score is %s" % (epoch, baby, baby_score))  # 打印过程信息
			
		# #########################
		# choose the best solution
		best_indv, max_score = max(chromo_score, key=lambda x : x[1])
		best_indv = self.decode_chromo(best_indv) # 将染色体信息解码成个体的二进制数组
		print("The best individual's code is %s and its score is %f" % (best_indv, max_score))  # 打印结果
		return best_indv