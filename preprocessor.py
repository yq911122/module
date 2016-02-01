
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
		l = (l-l.min())/(l.max()-l.min())*scale
		return l.round(decimals=3)
	return 'please numerize the data first!'


def small_var(df, thresh = 0.005):
	var = df.apply(lambda x: (x - x.min())/(x.max()-x.min())).var(axis=0).fillna(0.0)
	return list(var[var < thresh].index)

def numerize(l):
	dist = l.unique()
	ref = dict(zip(dist,range(len(dist))))
	l = l.map(ref)
	return l, ref

def discretize_df(df, cols, break_pts_lists):
	bin_list = [df]
	for break_pts, col in zip(break_pts_lists, cols):
		bin_list.extend([discretize(break_pts, df[col])])
	df = pd.concat(bin_list, axis=1)
	df = df.drop(labels = cols, axis=1)
	return df


def discretize(break_pts, l):
	break_pts = sorted(break_pts)
	names = [l.name + str(e) for e in break_pts]
	df = pd.DataFrame()
	for i in range(len(break_pts)):
		if i == 0: df[names[i]] = l.map(lambda x: int(x < break_pts[i])); continue
		df[names[i]] = l.map(lambda x: int(break_pts[i-1] <= x < break_pts[i]))
	# print df.head()
	return df

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