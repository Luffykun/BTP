import csv
import numpy as np
import pandas
import pandas as pd
from softmax import *
def frog_data_load():
	df = pd.read_csv("Datasets/Frogs_MFCCs.csv")
	df["Class"] = df["Family"] + " " + df["Genus"] + " " + df["Species"]
	y = df.values[:,-1]
	X = np.array(df.values[:,:22], dtype = float)
	return X, y

if __name__ == '__main__':
	softmax(*(frog_data_load()))
