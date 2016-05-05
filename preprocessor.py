import numpy as np

def shuffle(df):
	return df.reindex(np.random.permutation(df.index))

def __detect_outliers(l, upper=None, lower=None):
	if upper and lower: outliers = not (lower < l < upper)
	elif upper: outliers = l >= upper
	elif lower: outliers = l <= lower
	else: print "please provide upper bound or lower bound or both."; return None
	return outliers

def del_outliers(l, upper=None, lower=None):
	outliers = __detect_outliers(l, upper, lower)
	return l[not outliers]


def outliers_to_mean(l, upper=None, lower=None):
	outliers = __detect_outliers(l, upper, lower)
	l[outliers] = l[not outliers].mean()
	return l


def scale_cols(l, scale=1.0):
	if l.dtype.name != 'object': 
		l = (l-l.mean())/l.std()*scale
		return l.round(decimals=3)
	return 'please numerize the data first!'


def small_var(df, thresh = 0.005):
	var = df.apply(lambda x: (x - x.min())/(x.max()-x.min())).var(axis=0).fillna(0.0)
	# print var
	return list(var[var < thresh].index)

def numerize(l,getref=False):
	dist = l.unique()
	ref = dict(zip(dist,range(len(dist))))
	l = l.map(ref)
	if getref:	return l, ref
	return l

def discretize_df(df, cols, break_pts_lists):
	m = df[cols].copy()
	for break_pts, col in zip(break_pts_lists, cols):
		m[col] = discretize(break_pts, m[col])
	return m

def discretize(break_pts, l):
	m = l.copy()
	if break_pts == []: 
		m = 1
		return m
	break_pts = sorted(break_pts)
	discret_vals = range(1,len(break_pts)+2)

	re = m.index
	m = m.sort_values()
	for i in range(len(break_pts)+1):
		if i == 0: 
			m.iloc[0:break_pts[i]] = discret_vals[i]
		elif i == len(break_pts): 
			m.iloc[break_pts[i-1]:] = discret_vals[i]
		else: 
			m.iloc[break_pts[i-1]:break_pts[i]] = discret_vals[i]
	return m.reindex(re)


# def discretize(break_pts, l):
# 	if break_pts == []: return l
# 	break_pts = sorted(break_pts)
# 	names = [l.name + str(e) for e in break_pts]
# 	df = pd.DataFrame()
# 	for i in range(len(break_pts)):
# 		if i == 0: df[names[i]] = l.map(lambda x: int(x < break_pts[i])); continue
# 		df[names[i]] = l.map(lambda x: int(break_pts[i-1] <= x < break_pts[i]))
# 	# print df.head()
# 	return df

def get_correlated_cols(df, thr):
	cor = np.corrcoef(df.T)
	np.fill_diagonal(cor,0)
	cor_indice = np.where(cor>thr)
	cor_indice = [(i,j) for i, j in zip(cor_indice[0], cor_indice[1])]
	cor_cols = []
	for (i,j) in cor_indice:
		if (j,i) not in cor_cols:
			cor_cols.append((df.columns[i], df.columns[j]))
	return cor_cols


# def binarize_df():
# 	pass

# def binarize(l):
# 	for colName in colNames:
# 		ref = .ref[colName]
# 		.ref[colName] = []
# 		for k,v in ref.iteritems():
# 			l = .X[colName].map(lambda x: int(x == v))
# 			if sum(l.unique()) == 0: continue
# 			.X[k] = l
# 			.ref[colName].extend([k])
# 		dropKey = .ref[colName][0]
# 		# print dropKey
# 		.ref[colName] = []
# 		# .drop([colName,dropKey], axis=1, inplace=True)