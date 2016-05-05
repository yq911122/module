import pandas as pd
import numpy as np

from math import log
from entropy import ent, prob

def slice_ent(s):
	counts = np.bincount(s)
	vals = np.true_divide(counts, s.shape[0])
	return ent(vals), np.sum(vals != 0)

# @profile
def ent_eval(s, t):
	"""
	s: data set, pd.Series of a column as label
	t: cut point, int as index

	return: evaluate value
	"""
	size = float(s.shape[0])
	assert(0 <= t < size),"invalid cut point"
	s1, s2 = s[:t], s[t+1:]

	return (s1.shape[0]*slice_ent(s1)[0] + s2.shape[0]*slice_ent(s2)[0])/size
	# return s1.shape[0]/size*ent(s1)+s2.shape[0]/size*ent(s2)


def optimal_cut(s, eval_func=ent_eval):
	"""
	s: pd.Series of a column as label
	eval: function, evaluation method

	return: the optimal cut of the feature for discretization and related entropy
	"""

	min_entropy = np.inf
	optimal = 0

	for i in range(1, s.shape[0]):
		if s[i] == s[i-1]: continue
		curr_entropy = eval_func(s, i)
		if min_entropy > curr_entropy:
			min_entropy = curr_entropy
			optimal = i
	return optimal, min_entropy

def mdlp_accept(s, e, t):	
	N = s.shape[0]
	s1, s2 = s[:t], s[t+1:]
	# c1, c2 = s1.unique().shape[0], s2.unique().shape[0]	
	ent_s, c = slice_ent(s)
	ent_s1, c1 = slice_ent(s1)
	ent_s2, c2 = slice_ent(s2)

	gain = ent_s - e
	delta = log(pow(3,c)-2,2)-(c*ent_s-c1*ent_s1-c2*ent_s2)
	return gain > (log(N-1,2) + delta) / float(N)

def always_cut(s, e, t):
	return True

def EntropyDiscretizer(f, c, accept_strategy=mdlp_accept):
	"""
	f: feature, pd.Series
	c: related label, pd.Series
	
	f.sort_values(inplace=True)
	c = c.reindex(f.index)
	
	return: list of breaking points, l
	"""
	whole_ent = ent(c)
	if whole_ent == 0: return True
	t, e = optimal_cut(c)
	if accept_strategy(c, e, t):
		EntropyDiscretizer.cuts.append(t)
		if t <= EntropyDiscretizer.min_size  or t >= c.shape[0] - EntropyDiscretizer.min_size:
			return True

		c1, f1 = c[:t], f[:t]
		c2, f2 = c[t+1:], f[t+1:]
		EntropyDiscretizer(f1, c1, accept_strategy)
		EntropyDiscretizer(f2, c2, accept_strategy)
	return True
EntropyDiscretizer.cuts = []
EntropyDiscretizer.min_size = 100

def EntropyDiscretize(X, y, accept_strategy=mdlp_accept):
	c = y.copy()
	df_cuts = []
	for i in range(X.shape[1]):
		f = X[:,i]
		order = np.argsort(f)
		EntropyDiscretizer(f[order], c[order], accept_strategy)
		df_cuts.append(EntropyDiscretizer.cuts)
		EntropyDiscretizer.cuts = []
	return df_cuts


def main():
	df = pd.read_csv('../bnp-paribas-cardif-claims-management/data/train11.csv', index_col=0)
	X, y = df[[e for e in df.columns if e != 'target']].values, df['target'].astype(np.int32).values
	a = EntropyDiscretize(X, y, always_cut)
	print a
	text_file = open("Output.txt", "w")
	text_file.write(str(a))
	text_file.close()

if __name__ == '__main__':
	main()