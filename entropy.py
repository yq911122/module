import pandas as pd
import numpy as np

from math import log

import scipy.stats

def prob(x):
	# unique, counts = np.unique(x, return_counts=True)

	return x.value_counts()/x.count()

def cond_prob(x, y):
	"""
	x: pd.Series, variable for conditional probability calculation
	y: pd.Series, conditional variable

	return: conditional probability of X against y
	"""
	return x.groupby(y).apply(prob)

def joint_prob(x, y):
	"""
	:type x: pd.Series
	:type y: pd.Series
	:rtype: pd.DataFrame
	"""
	# val = pd.
	py = prob(y)
	# print py
	grouped = x.groupby(y)
	# p = []
	# for name, group in grouped:
	# 	p.append(prob(group)*py[name])
	# return pd.concat(p)
	return x.groupby(y).apply(lambda e: prob(e)*py[e.name])


def ent(px):
	return scipy.stats.entropy(px, base=2)

def ent2(px):
	"""
	x: pd.Series, probability distribution for entropy calculation

	return: entropy of x. ent(x) = -\sum_iP(x_i)log_2(P(x_i))
	"""
	return -px.apply(lambda e: e*log(e, 2)).sum()

# def relative_ent2(x, y):
# 	pxy = joint_prob(x, y)
# 	py = prob(y)
# 	pxy.index = pxy.index.droplevel(1)
# 	tmp = pd.DataFrame(pxy).join(py)
# 	tmp.columns = ['joint','ind']
# 	# print tmp.head()
# 	return scipy.stats.entropy(tmp['joint'], tmp['ind'],2)

def relative_ent(x, y):
	"""
	x: pd.Series, data for entropy calculation
	y: pd.Series, conditional variable

	return: relative entropy of X over y. cond_ent(x, y) = -\sum_jP(y_j)\sum_iP(x_i|y_j)log_2(P(x_i|y_j))
	"""
	py = prob(y).to_dict()
	cond_pxy = cond_prob(x, y)
	temp = {}

	for y_i, x_yi in cond_pxy.groupby(level=0):
		temp[y_i] = x_yi.apply(lambda e: e*log(e, 2)).sum()

	return -sum([py[k]*temp[k] for k in py.keys()])

def infoGain(x, y):
	return ent(prob(x)) - relative_ent(x, y)

def main():
	df = pd.read_csv('./Lung-cancer_proc.csv', index_col=0)
	x, y = df[df.columns[2]], df[df.columns[0]].astype(np.int32)
	# print x.head()
	# print y.head()
	# # print X.value_counts()
	# # print y.value_counts()

	i = 0
	while i < 100000:
		np.bincount(y)
		y.value_counts()
		# ent(x)
		# ent2(x)
		i += 1
	# print relative_ent(x, y)
	# print relative_ent2(x, y)
	# print infoGain(x, y)
	# print infoGain(y, x)
	# g = x.groupby(y).apply(lambda e: e.index[0])
	# print g
if __name__ == '__main__':
	main()