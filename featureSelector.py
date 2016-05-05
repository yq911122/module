import pandas as pd
import numpy as np

class featureSelector(object):
	"""docstring for featureSelector"""
	def __init__(self, X, y):
		super(featureSelector, self).__init__()
		self.X = X
		self.y = y
		

	def process(self):
		"""
		X: features instances, pd.DataFrame
		y: class instances, pd.Series
		eval: return wheather feature(s) meet the standard
		search: return subsets for evaluation

		return: X after processing
		"""

		while not self.stop():
			self.cand = self.search(self.X,self.y)
			self.eval(self.cand)
		return self.validate()

	def eval(self, cand):
		pass

	def search(self, X, y):
		pass

	def stop(self):
		pass

	def validate(self):
		pass


class simpleCorrSelector(featureSelector):
	"""docstring for simpleCorrSelector"""
	def __init__(self, X, y, cor, thr):
		super(simpleCorrSelector, self).__init__(X, y)
		self.X = X
		self.y = y
		self.cor = cor
		self.c = 0
		self.w = [0]*X.shape[1]
		self.thr = thr

	def eval(self,cand):
		self.w[self.c-1] = self.cor(cand, self.y)

	def search(self, X, y):
		cand = None
		if self.c < self.X.shape[1]:
			cand = X.iloc[:,self.c]
			self.c += 1
		return cand

	def stop(self):
		return self.c >= self.X.shape[1]

	def validate(self):
		return self.post_process()	

	def post_process(self, sort=True):
		# print self.w
		features = [i for i in range(len(self.w)) if self.w[i] > self.thr]
		self.w = [e for e in self.w if e > self.thr]
		if sort:
			features = [x for y, x in sorted(zip(self.w, features),reverse=True)]
			self.w = sorted(self.w, reverse=True)
		features = [self.X.columns[e] for e in features]
		print features
		print self.w
 		return self.X[features], self.w
		


class corrSelector(featureSelector):
	"""docstring for corrSelector"""
	def __init__(self, X, y, cor, thr):
		super(corrSelector, self).__init__(X, y)
		self.cor = cor
		fs = simpleCorrSelector(X, y, cor, thr)
		self.X, _ = fs.process()
		self.f1 = self.X.iloc[:,0]
		self.p = 0


	def eval(self, cand):
		if cand is not None:
			fq = self.X.iloc[:,cand]
			while fq is not None:
				print self.f1.name
				print fq.name
				a = self.cor(self.f1, fq)
				b = self.cor(fq, self.y)
				if a >= b:
					print a
					print b
				# if self.cor(self.f1, fq) >= self.cor(fq, self.y):
					self.X.drop(self.X.columns[cand], axis=1, inplace=True)
				else: cand += 1
				print "\n"	

				if cand >= self.X.shape[1]:	break
				fq = self.X.iloc[:,cand]
			if self.p < self.X.shape[1]:
				self.f1 = self.X.iloc[:,self.p]

	def search(self, X, y):
		self.p += 1
		# print self.p
		if self.p >= X.shape[1]:
			return None
		return self.p

	def stop(self):
		return self.p >= self.X.shape[1]

	def validate(self):
		return self.X



from scipy.stats import pearsonr
def pearson(x, y):
	return pearsonr(x, y)[0]

from entropy import infoGain, ent
def symmetricalUncertainty(x, y):
	return 2*infoGain(x, y)/(ent(x)+ent(y))

import preprocessor as proc
import discretizer as dt

def main():
	df = pd.read_csv('../bnp-paribas-cardif-claims-management/data/train21.csv',index_col=0)
	x_cols = [e for e in df.columns if e != 'target']
	con_cols = ['v10','v14','v22','v34','v40','v114']
	p = [[82493, 35206, 1688, 1016, 823, 74, 129, 208, 201, 414, 20418, 15825, 5488, 2534, 962, 5, 152, 120, 401, 285, 281, 2, 355, 142, 21, 22, 8, 2638, 2011, 832, 821, 101, 71, 7, 431, 298, 18, 11, 65, 36, 3719, 3705, 4525, 4041, 3950, 6, 1817, 1810, 7, 4557, 11306, 11292, 5, 16165, 1767, 10, 39, 14252, 14181, 12136, 8349, 3448, 2051, 157, 124, 387, 192, 155, 5, 431, 10, 45, 1383, 4159, 414, 400, 341, 12, 1193, 419, 2, 55, 315, 238, 637, 410, 399, 202, 27, 6, 619, 18, 71, 4580, 477, 185, 180, 8, 101, 76, 573, 547, 12, 74, 12941, 6500, 248, 206, 2046, 464, 67, 1557, 1071, 1003, 689, 602, 59, 3265, 735, 134, 118, 452, 443, 31, 36, 55, 1692, 581, 270, 191, 6, 102, 0, 136, 99, 828, 1318, 210, 183, 425, 215, 184, 7, 13, 4802, 700, 668, 20, 1204, 815, 170, 121, 529, 494, 15, 112, 92, 21, 5125, 5047, 619, 90, 2959, 2920, 518, 131, 75, 78, 33], [95977, 66394, 82, 27330, 646, 331, 6, 28, 14478, 1546, 866, 835, 13, 8232, 7958, 48, 150, 13, 76, 2989, 2979, 77, 12137, 1743, 1722, 55, 4157, 1097, 24, 236, 175, 358, 148, 126, 204, 144, 126, 2296, 246, 135, 118, 50, 11914, 7291, 7268, 141, 108, 154, 131, 4026, 3965, 168, 38, 33, 87], [69465, 4006, 482, 141, 124, 278, 25, 13521, 13038, 13019, 97, 979, 779, 178, 165, 18, 188, 21355, 21068, 232, 2, 209, 195, 20616, 10, 208, 189, 1780, 1675, 986, 865, 859, 116, 548, 491, 88, 13, 11, 40834, 40813, 35], [89593, 20020, 13800, 13581, 11927, 11900, 704, 683, 27, 33, 186, 172, 5502, 5494, 20, 19459, 11, 32505, 32486, 74, 5395, 5362, 19292], [12862, 1456, 1370, 6928, 6736, 10, 98, 4375, 4005, 3590, 635, 5, 759, 10, 648, 589, 40, 18, 288, 89, 92678, 11121, 906, 890, 167, 2, 393, 45, 77, 67127, 13658, 31, 10, 14364, 8704], [97254, 16956, 4095, 1494, 118, 111, 48, 8, 3112, 15, 894, 848, 7611, 5062, 4880, 4872, 22, 2540, 74, 27004, 26844, 14181, 12872, 11761, 8, 486, 464, 128, 120, 34, 2, 643, 44, 11993, 19, 285, 273, 42037, 41916, 34321, 24087, 23973, 21966, 21887, 490, 13, 1480, 14, 1086, 944, 900, 124, 29, 11, 43, 26, 17020]]


	df[con_cols] = proc.discretize_df(df, con_cols, p)

	# # print df.columns
	X, y = df[x_cols], df['target']
	# cuts = dt.EntropyDiscretize(X, y)
	# print cuts
	


	featureSelector = corrSelector(X, y, symmetricalUncertainty, 0.0)
	sub_X = featureSelector.process()
	print sub_X.columns



if __name__ == '__main__':
	main()



	# class relief(featureSelector):
# 	"""docstring for relief"""
# 	def __init__(self, X, y, thr, sample_frac):
# 		super(relief, self).__init__(X, y)
# 		self.w = [0]*X.shape[1]
# 		self.size = sample_frac*float(X.shape[0])
# 		self.thr = thr
# 		self.c = 0
# 		grp = self.X.groupby(self.y)
# 		self.y1, self.y2 = y.unique()[0], y.unique()[1]
# 		self.X1, self.X2 = grp.get_group(self.y1), grp.get_group(self.y2)

# 	def eval(self, x, y):

# 		d1, d2 = near(x, y)
# 		self.updatew(x, d1, d2)

# 	def updatew(self, x, hit, miss):
# 		diff1 = distance(x, hit)
# 		diff2 = distance(x, miss)
# 		self.w = [e-p+q for e, p, q in zip(self.w, diff1, diff2)]

# 	def search(self, X, y):
# 		X_sample = X.sample(1)
# 		y_sample = y[X_sample.index]
# 		self.eval(X_sample, y_sample)

# 	def stop(self):
# 		self.c += 1
# 		return self.c == self.size

# 	def near(self, x, y):
# 		X1, X2 = self.X1, self.X2
# 		a1, a2 = -1.0, -1.0
# 		if y == self.y1:
# 			X1 = self.X1[self.X1!=x]
# 			a1 = 1.0
# 		else:
# 			X2 = self.X2[self.X2!=x]
# 			a2 = 1.0

# 		#waiting...


	
# 	def post_process(self):
# 		features = [i for i in range(len(self.w)) if self.w[i] > self.thr]
# 		return self.X[features]