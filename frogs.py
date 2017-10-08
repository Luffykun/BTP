import csv
import numpy as np
import pandas
import pandas as pd
from softmax import *
df = pd.read_csv("Datasets/Frogs_MFCCs.csv")
df["Class"] = df["Family"] + " " + df["Genus"] + " " + df["Species"]
y = df.values[:,-1]
X = np.array(df.values[:,:22], dtype = float)
w, error = softmax(X,y)
