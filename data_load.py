import csv
import timeit
import numpy as np
import pandas
import pandas as pd
from softmax import *
import plots
def frogs_data_load():
	df = pd.read_csv("Datasets/Frogs_MFCCs.csv")
	df["Class"] = df["Family"] + " " + df["Genus"] + " " + df["Species"]
	y = df.values[:,-1]
	X = np.array(df.values[:,:22], dtype = float)
	return X, y

def faults_data_load():
	df = pd.read_csv("Datasets/Faults.NNA", sep = "\t", header=None)
	df = df[df[33] == 0]
	y = np.argmax(df.values[:,27:], axis=1)
	X = df.values[:, :27]
	X = X/np.max(np.abs(X), axis=0)
	return X, y

def statlog_data_load_notime():
	df = pd.read_csv("Datasets/shuttle.trn", sep="\t", header=None)
	y = df.values[:,-1]
	X = df.values[:,:-1]
	X, y = getNewBatch(X, y, 5000)
	print X.shape, y.shape
	X = X/np.max(np.abs(X), axis=0)
	return X, y

def sensorless_drive_data_load():
	df = pd.read_csv("Datasets/Sensorless_drive_diagnosis.txt", sep="\t", header=None)
	y = df.values[:,-1]
	X = df.values[:,:-1]
	X = X/np.max(np.abs(X), axis=0)
	return X, y

def wine_data_load():
	df = pd.read_csv("Datasets/wine.data", header=None)
	y = df.values[:,0]
	X = df.values[:,1:]
	X = X/np.max(X, axis=0)
	return X, y

def glass_data_load():
	df = pd.read_csv("Datasets/glass.data", header=None)
	y = df.values[:,-1]
	X = df.values[:,1:-1]
	X = X/np.max(X, axis=0)
	return X, y

def iris_data_load():
	df = pd.read_csv("Datasets/iris.data")
	y = df.values[:,-1]
	X = np.array(df.values[:,:-1], dtype=float)
	X = X/np.max(X, axis=0)
	return X, y

def satimage_data_load():
	df = pd.read_csv("Datasets/satimage/sat.trn", delimiter="\s")
	y = df.values[:,-1]
	X = df.values[:,:-1]
	return X, y

X, y = faults_data_load()
if __name__ == '__main__':
	print "Running softmax..."
	#t = timeit.Timer(lambda: softmax(X, y, threshold=1e-4,stochastic=True, batch_size=128))
	#print t.timeit(number = 1)
	#t = timeit.Timer(lambda: softmax(X, y, threshold=1e-4,stochastic=True, batch_size=32))
	#print t.timeit(number = 1)
	#t = timeit.Timer(lambda: softmax(X, y, threshold=1e-4,stochastic=True, batch_size=256))
	#print t.timeit(number = 1)
	#t = timeit.Timer(lambda: softmax(X, y, threshold=1e-4,stochastic=True, batch_size=64))
	#print t.timeit(number = 1)
	t = timeit.Timer(lambda: softmax_bfgs(X, y))
	print t.timeit(number = 1)
	print "Softmax run over."
	print "Running SVM..."
	t = timeit.Timer(lambda: train_svm(X, y))
	print t.timeit(number = 1)
	#print "SVM run over."
	print "Running one vs one..."
	t = timeit.Timer(lambda: one_vs_one(X, y, stochastic=False, batch_size=32))
	print t.timeit(number = 1)
	print "one vs one run over."
	print "Running one vs all..."
	t = timeit.Timer(lambda: one_vs_all(X, y, stochastic=False, batch_size=128))
	print t.timeit(number = 1)
	print "one vs all over."
