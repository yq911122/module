
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
def cross_validation(clf, X, Y, cv=5):
	'''
	:clf : classifier with fit() and predict() method
	:X : pd.DataFrame; features
	:Y : pd.DataFrame(1 column) or pd.Series; labels
	:cv : int; cross validation folders

	:return : list of float; cross validation scores
	'''

	k = [(len(X)-1)/cv*j for j in range(cv+1)]
	score = [0.0]*cv
	for i in range(cv):		
		train_x, train_y = pd.concat([X.loc[:k[i],:],X.loc[k[i+1]:,:]]), pd.concat([Y.loc[:k[i]],Y.loc[k[i+1]:]])
		test_x, test_y = X.loc[k[i]:k[i+1],:], Y.loc[k[i]:k[i+1]]
		model = clf.fit(X,Y)
		pred = clf.predict(test_x)
		score[i] = (pred == test[y_name]).sum()/float(len(test))
	return score