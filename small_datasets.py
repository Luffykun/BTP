import csv
import plots
import numpy as np
import pandas as pd

def wine_data_load():
	df = pd.read_csv("Datasets/wine.data")
	y = df.values[:,0]
	X = df.values[:,1:]
	return X, y

def glass_data_load():
	df = pd.read_csv("Datasets/glass.data")
	y = df.values[:,-1]
	X = df.values[:,1:-1]
	return X, y

def iris_data_load():
	df = pd.read_csv("Datasets/iris.data")
	y = df.values[:,-1]
	X = np.array(df.values[:,:-1], dtype=float)
	return X, y

def satimage_data_load():
	df = pd.read_csv("Datasets/satimage/sat.trn", delimiter="\s")
	y = df.values[:,-1]
	X = df.values[:,:-1]
	return X, y

if __name__ == "__main__":
	plots.plot_classes(*iris_data_load())
	plots.plot_classes(*wine_data_load())
	plots.plot_classes(*glass_data_load())
	plots.plot_classes(*satimage_data_load())
