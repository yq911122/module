
def cvScore(clf, X, Y):
	from sklearn import cross_validation
	return cross_validation.cross_val_score(clf, X, Y, cv=5)

def paramSelector(clf, params, X, Y):
	from sklearn.grid_search import GridSearchCV
	# params = {"n_estimators": [10, 50, 100], "min_samples_leaf": [5, 15, 30]}
	clf = GridSearchCV(clf, params)
	clf.fit(X, Y)
	return clf.best_estimator_, clf.grid_scores_