import imp

def load(name):
	path = 'D:/file_data/data science/projects/Module/' + name +'.py'
	return imp.load_source(name, path)
