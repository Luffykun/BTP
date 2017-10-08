# coding: utf-8
get_ipython().magic(u'edit softmax.py')
import * from softmax
from softmax import *
X = np.arange(150)
X
X2 = np.ones([150])*150-X
X2
X2[:75] += np.random.rand(75)*10
X2[75:] -= np.random.rand(75)*10
y = np.ones([150])
y[75:]=2
X = np.columns_stack((X,X1))
X = np.column_stack((X,X1))
X = np.column_stack((X,X2))
get_ipython().magic(u'save data_gen1 4 6 8-11 14')
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
softmax(X,y)
w
w = np.random.rand([2,3])
w = np.random.rand(2,3)*10
w = np.random.rand(2,3)*20 - np.ones([2,3])*10
X.dot(w.t)
X.dot(w.T)
X_pad = np.pad(X, ((0,0),(1,0)), 'constant', constant_values=1)
X.dot(w.T)
X_pad.dot(w.T)
X_dum = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
X_dum.dot(w.T)
X_dum_pad = np.pad(X_dum, ((0,0),(1,0)), 'constant', constant_values=1)
X_dum_pad.dot(w.T)
np.exp(X_dum_pad.dot(w.T))
np.exp(X_pad.dot(w.T))
np.log(np.sum(np.exp(X_pad.dot(w.T)), axis = 1))
np.log(np.sum(np.exp(X_pad.dot(w.T) - np.max(X_pad.dot(w.T))), axis = 1))
np.log(np.sum(np.exp(X_pad.dot(w.T) - np.max(X_pad.dot(w.T))), axis = 1)) + np.max(X_pad.dot(w.T))
get_ipython().magic(u'edit softmax.py')
get_ipython().magic(u'edit softmax.py')
get_ipython().magic(u'edit softmax.py')
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
X_pad.shape
np.max(X_pad.dot(w.T), axis = 1).shape
np.log(np.sum(np.exp(X_pad.dot(w.T) - np.max(X_pad.dot(w.T), axis = 1)), axis = 1))
np.log(np.sum(np.exp(X_pad.dot(w.T) - np.max(X_pad.dot(w.T), axis = 1, keepdims = True)), axis = 1))
np.log(np.sum(np.exp(X_pad.dot(w.T) - np.max(X_pad.dot(w.T), axis = 1, keepdims = True)), axis = 1)) + np.max(X_pad.dot(w.T), axis = 1)
(np.log(np.sum(np.exp(X_pad.dot(w.T) - np.max(X_pad.dot(w.T), axis = 1, keepdims = True)), axis = 1)) + np.max(X_pad.dot(w.T), axis = 1)).shape
np.max(X_pad.dot(w.T), axis = 1).shape
np.log(np.sum(np.exp(X_pad.dot(w.T) - np.max(X_pad.dot(w.T), axis = 1, keepdims = True)), axis = 1)) + np.max(X_pad.dot(w.T), axis = 1, keepdims = True)
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
a = np.random.rand(50)
a = np.log(a)
a
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
softmax(X,y)
w
classes = np.unique(y)
classes_expanded = np.matlib.repmat(classes, m, 1)
classes_expanded = np.matlib.repmat(classes, 150, 1)
y = np.array(classes_expanded == np.column_stack((y,)), dtype = float)
evalLoss(X_pad, y, w)
get_ipython().magic(u'debug evalLoss(X_pad, y, w)')
get_ipython().magic(u'debug evalLoss(X_pad, y, w)')
get_ipython().magic(u'pdb evalLoss(X_pad, y, w)')
get_ipython().magic(u'run -d evalLoss(X_pad, y, w)')
get_ipython().magic(u'run data_gen1.py')
get_ipython().magic(u'edit data_gen1.py')
get_ipython().magic(u'run data_gen1.py')
classes = np.unique(y)
classes_expanded = np.matlib.repmat(classes, 150, 1)
y_pad = np.array(classes_expanded == np.column_stack((y,)), dtype = float)
temp1 = np.max(X_pad.(w.T), axis = 1)
temp1 = np.max(X_pad.dot(w.T), axis = 1)
temp1
temp1.shape
temp1 = np.max(X_pad.dot(w.T), axis = 1, keepdims = True)
temp2 = X_pad.dot(w.T)
temp2
a = np.exp(temp2 - temp1)
a
np.sum(a, axis = 1)
b = np.sum(a, axis = 1)
np.log(b)
y.dot(w)
y_pad.dot(w)
X_pad*(y_pad.dot(w))
np.sum(X_pad*(y_pad.dot(w)), axis = 1)
np.sum(X_pad*(y_pad.dot(w)), axis = 1).shape
X_pad.dot(w.T) - np.sum(X_pad*(y_pad.dot(w)), axis = 1).shape
X_pad.dot(w.T) - np.sum(X_pad*(y_pad.dot(w)), axis = 1)
get_ipython().magic(u'edit softmax.py')
X_pad.dot(w.T) - np.sum(X_pad*(y_pad.dot(w)), axis = 1, keepdims = True)
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
evalLoss(X,y,w)
evalLoss(X_pad,y_pad,w)
evalLoss(X_pad,y_pad,descent(X,y,w,2))
evalLoss(X_pad,y_pad,descent(X_pad,y_pad,w,2))
evalLoss(X_pad,y_pad,descent(X,y,w,2))
evalLoss(X_pad,y_pad,w)
evalLoss(X_pad,y_pad,descent(X_pad,y_pad,w,2))
w2 = descent(X,y,w,2)
w2 = descent(X_pad,y_pad,w,2)
evalLoss(X_pad,y_pad,descent(X_pad,y_pad,w,2))
evalLoss(X_pad,y_pad,descent(X_pad,y_pad,w2,2))
get_ipython().magic(u'edit softmax.py')
softmax(X,y,w)
softmax(X,y,w)
softmax(X,y)
w_final, error = softmax(X,y)
w_final, error = softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
w_ideal = np.array([[155, 2, 2],[5, 3, 3]])
evalLoss(X,y,w_ideal)
evalLoss(X_pad,y_pad,w_ideal)
evalLoss(X_pad,y_pad,-w_ideal)
w_eh = np.array([[62, 2463, 8271],[69, 8207, 2422]], dtype = float)
w_ideal = np.array([[155, 2, 2],[5, 3, 3]], dtype = float)
evalLoss(X_pad,y_pad,-w_ideal)
evalLoss(X_pad,y_pad,w_ideal)
evalLoss(X_pad,y_pad,w_eh)
import matplotlib.pyplot as plt
plt.plot(X[:75, 0],X[:75,1],'ro')
plt.show()
plt.plot(X[:75, 0],X[:75,1],'ro', X[75:, 0], X[75:,1], 'bo')
plt.show()
plt.show()
plt.plot(X[:75, 0],X[:75,1],'ro', X[75:, 0], X[75:,1], 'bo')
plt.show()
w_eh
w_eh_line = w_eh[0,:] - w_eh[1,:]
w_eh_line
w_ideal_line = w_ideal[0,:] - w_ideal[1,:]
w_ideal_line
x = np.arange(150)
x
x = np.arange(160)
x = np.arange(150)
y_ideal = np.ones([150])*150 - np.arange(150)
y_ideal
w_eh_line
w_eh_line = w_eh_line/w_eh_line[2]
w_eh_line
y_eh = np.ones([150])*(-w_eh_line[0]) - np.arange(150)*(-w_eh_line[1])
plt.plot(X[:75, 0],X[:75,1],'ro', X[75:, 0], X[75:,1], 'bo', x, y_ideal, 'g', x, y_eh)
plt.show()
y_eh = np.ones([150])*(w_eh_line[0]) + np.arange(150)*(-w_eh_line[1])
plt.plot(X[:75, 0],X[:75,1],'ro', X[75:, 0], X[75:,1], 'bo', x, y_ideal, 'g', x, y_eh)
plt.show()
X1 = np.random.rand(150)*200 - np.ones([150])*100
X2 = np.random.rand(150)*200 - np.ones([150])*100
plt.plot(X1, X2, 'ro')
plt.show()
np.where(X1+X2 > 0)
np.where(X1+X2 >= 0)
np.where(X1+X2 <= 0)
y = np.ones([150])
y[np.where(X1+X2 < 0)] = 2
2
y
plt.plot(X1[np.where(X1+X2>=0)], X2[np.where(X1+X2>=0)], 'ro')
plt.show()
X = np.column_stack((X1,X2))
softmax(X,y)
softmax(X,y)
softmax(X,y)
w_hmm = np.array([[704, 24187, 21697],[822, -26215, -29209]])
w_hmm_line = w_hmm[0] - w_hmm[1]
w_hmm_line
w_hmm_line /= w_hmm_line[3]
w_hmm_line /= w_hmm_line[2]
w_hmm_line
w_hmm = np.array([[704, 24187, 21697],[822, -26215, -29209]], dtype = float)
w_hmm_line = w_hmm[0] - w_hmm[1]
w_hmm_line /= w_hmm_line[2]
w_hmm_line
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
X_pad = np.pad(X, ((0,0),(1,0)), 'constant', constant_values=1)
classes = np.unique(y);classes_expanded = np.matlib.repmat(classes, m, 1)
classes = np.unique(y);classes_expanded = np.matlib.repmat(classes, 150, 1)
y_pad = np.array(classes_expanded == np.column_stack((y,)), dtype = float)
get_ipython().magic(u'edit softmax.py')
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
get_ipython().magic(u'edit softmax.py')
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
softmax(X,y)
get_ipython().magic(u'timeit softmax(X,y)')
softmax(X,y)
get_ipython().magic(u'run data_gen1.py')
softmax(X,y)
