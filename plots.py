import softmax
from softmax import *
import matplotlib.pyplot as plt

def pca(data, num_components):
	X = data - np.mean(data, axis=0)
	devs = X.T.dot(X)/X.shape[0]
	eig, eigv = np.linalg.eigh(devs)
	pcas = eigv[:,-num_components:]
	Y = pcas.T.dot(X.T).T
	X_new = pcas.dot(Y.T).T
	data_new = X_new + np.mean(data, axis=0)
	return Y

def plot_classes(X, y):
	X = pca(X, 2)
	w = softmax(X, y)["model"]
	classes = np.unique(y)
	figure, ax = plt.subplots()
	for a_class in classes:
		#figure, ax = plt.subplots()
		instances = X[y == a_class]
		ax.scatter(instances[:,0], instances[:,1], label=a_class)
	ax.legend()
	plt.show()
