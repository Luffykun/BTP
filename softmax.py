import numpy as np
import numpy.matlib
from scipy.optimize import minimize

def pad(X):
	return np.pad(X, ((0,0),(1,0)), 'constant', constant_values=1)

def encode(y):
	classes = np.unique(y)
	classes_expanded = np.matlib.repmat(classes, y.shape[0], 1)
	return np.array(classes_expanded == np.column_stack((y,)), dtype = int)

def divide_dataset(X, y, ratio = 0.7):
	m = X.shape[0]
	d = X.shape[1]
	k = y.shape[1]
	X_aug = np.hstack((X,y))
	np.random.shuffle(X_aug)
	m_train = int(m*ratio)
	X_test = X_aug[m_train:,:d]
	y_test = X_aug[m_train:,d:]
	X_train = X_aug[:m_train,:d]
	y_train = X_aug[:m_train,d:]
	return X_train, y_train, X_test, y_test

def evalLoss(X, y, w):
	m = X.shape[0]
	temp = X.dot(w.T) - np.sum(X*(y.dot(w)), axis = 1, keepdims = True)
	temp2 = np.max(temp, axis = 1, keepdims = True)
	return (np.sum(np.log(np.sum(np.exp(temp - temp2), axis = 1)) + np.reshape(temp2, [-1])))/m

def evalGrad(X, y, w):
	m = X.shape[0]
	k = w.shape[0]
	temp = X.dot(w.T)
	return (1/np.sum(np.exp(np.reshape(temp, [m,k,1]) - np.reshape(temp, [m,1,k])), axis = 1) - y).T.dot(X)/m

def predict(X, w):
	return np.argmax(X.dot(w.T), axis = 1)

def secant(fun, IV1, IV2, max_iter=3, epsilon=1e-4):
	x1 = IV1
	x2 = IV2
	val1 = fun(x1)
	val2 = fun(x2)
	i = 0
	while val2 > epsilon and i < max_iter:
		temp = x2
		x2 = x2 - val2*(x2 - x1)/(val2 - val1)
		x1 = temp
		val1 = val2
		val2 = fun(x2)
	return x2

def getNewBatch(X, y, size):
	m = X.shape[0]
	d = X.shape[1]
	X_aug = np.hstack((X,y))
	np.random.shuffle(X_aug)
	start = np.random.randint(0,m-size)
	X_batch = X_aug[start:start+size,:d]
	y_batch = X_aug[start:start+size,d:]
	return X_batch, y_batch

def descent(X, y, w, stochastic = False, batch_size = 0):
	if stochastic:
		X, y = getNewBatch(X, y, batch_size)
	grad_w = evalGrad(X, y, w)
	#alpha = minimize(lambda alpha: evalLoss(X, y, w - alpha*grad_w), 0, method = 'BFGS', jac = lambda alpha: np.array([-1*np.sum(evalGrad(X, y, w - alpha*grad_w)*grad_w)])).x[0]
	#alpha = 2
	alpha = secant(lambda alpha: np.array([-1*np.sum(evalGrad(X, y, w - alpha*grad_w)*grad_w)]), 0, 2, 3)
	return w - alpha*grad_w

def evaluate(X, y, w, classes):
	print "Class", "True Labels", "Predicted Labels", "Precision", "Recall" 
	pred = classes[predict(X,w)]
	y_n = classes[np.argmax(y, axis = 1)]
	true_counts = dict(zip(*np.unique(y_n, return_counts = True)))
	pred_counts = dict(zip(*np.unique(pred, return_counts = True)))
	pos_counts = dict(zip(*np.unique(y_n[pred==y_n], return_counts = True)))
	for class_ex in classes:
		true = true_counts.get(class_ex, 0)
		predicted = pred_counts.get(class_ex, 0)
		pos = pos_counts.get(class_ex, 0)
		dist = dict(zip(*np.unique(pred[y_n==class_ex], return_counts = True)))
		print "%s %d %d %d %.2f %.2f" % (class_ex, true, predicted, pos, float(pos)/predicted, float(pos)/true)
	print encode(pred).T.dot(encode(y_n))

def softmax_bfgs(X, y, pad_ones=True, one_hot_encode=True):
	if pad_ones:
		X = pad(X)
	if one_hot_encode:
		y = encode(y)
	m = X.shape[0] #Number of data points
	d = X.shape[1] #Number of features
	k = y.shape[1]
	X, y, X_test, y_test = divide_dataset(X, y)
	w0 = (np.random.rand(k,d)*20 - np.ones((k,d))*10).flatten()
	print w0.shape
	model =  minimize(lambda w: evalLoss(X, y, np.reshape(w, (k,d))), w0, method = 'BFGS', jac = lambda w: evalGrad(X, y, np.reshape(w, (k,d))).flatten()).x
	model = np.reshape(model, (k,d))
	return model, evalLoss(X_test, y_test, model)

def softmax(X, y, pad_ones=True, one_hot_encode=True, stochastic=False, batch_size=0):
	"""Return a softmax model and the corresponding test error. The training:testing divide used is 70:30"""
	if pad_ones:
		X = pad(X)
	if one_hot_encode:
		classes = np.unique(y)
		y = encode(y)
	else:
		classes = np.arange(y.shape[1])
	
	m = X.shape[0] #Number of data points
	d = X.shape[1] #Number of features
	k = y.shape[1]
	X, y, X_test, y_test = divide_dataset(X, y)
	w = np.random.rand(k,d)*20 - np.ones((k,d))*10
	epsilon = 1e-5
	loss_new = evalLoss(X, y, w)
	loss_old = loss_new + 10**3
	while(abs(loss_old-loss_new) > epsilon):
		if stochastic:
			w = descent(X, y, w, stochastic=True, batch_size=batch_size)
		else:
			w = descent(X, y, w)
		loss_old = loss_new
		loss_new = evalLoss(X, y, w)
		#print loss_old, loss_new
	evaluate(X_test, y_test, w, classes)
	evaluate(np.vstack((X, X_test)), np.vstack((y, y_test)), w, classes)
	return w, evalLoss(X_test, y_test, w)
