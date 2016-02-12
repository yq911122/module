
def cvScore(clf, X, Y):
	from sklearn import cross_validation
	return cross_validation.cross_val_score(clf, X, Y, cv=5)

def paramSelector(clf, params, X, Y):
	from sklearn.grid_search import GridSearchCV
	# params = {"n_estimators": [10, 50, 100], "min_samples_leaf": [5, 15, 30]}
	clf = GridSearchCV(clf, params)
	clf.fit(X, Y)
	return clf.best_estimator_, clf.grid_scores_

#untest
def cross_validation(clf, X, Y, cv=5, avg=False):
	'''
	:clf : classifier with fit() and predict() method
	:X : pd.DataFrame; features
	:Y : pd.DataFrame(1 column) or pd.Series; labels
	:cv : int; cross validation folders

	:return : list of float; cross validation scores
	'''
	import pandas as pd

	k = [(len(X))/cv*j for j in range(cv+1)]
	# print len(X)
	score = [0.0]*cv
	for i in range(cv):	
		train_x, train_y = pd.concat([X[:k[i]],X[k[i+1]:]]), pd.concat([Y[:k[i]],Y[k[i+1]:]])
		test_x, test_y = X[k[i]:k[i+1]], Y[k[i]:k[i+1]]

		# print train_y
		# print len(test_x)
		clf.fit(X,Y)
		pred = clf.predict(test_x)
		# pred = [0]*len(test_y)
		score[i] = (pred == test_y).sum()/float(len(test_y))
	if avg: return sum(score)/float(len(score))
	return score