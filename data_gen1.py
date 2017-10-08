# coding: utf-8
import numpy as np
X = np.arange(150)
X2 = np.ones([150])*150-X
X2[:75] += np.random.rand(75)*10
X2[75:] -= np.random.rand(75)*10
y = np.ones([150])
y[75:]=2
X = np.column_stack((X,X2))
