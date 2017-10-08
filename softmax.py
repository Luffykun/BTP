import numpy as np
import numpy.matlib

def evalLoss(X, y, w):
	m = X.shape[0]
	temp = X.dot(w.T) - np.sum(X*(y.dot(w)), axis = 1, keepdims = True)
	temp2 = np.max(temp, axis = 1, keepdims = True)
	
	#return (np.sum(np.log(np.sum(np.exp(X.dot(w.T)), axis = 1))) - np.sum(X.dot((y.dot(w)).T)))/m
	return (np.sum(np.log(np.sum(np.exp(temp - temp2), axis = 1)) + np.reshape(temp2, [-1])))/m

def evalGrad(X, y, w):
	m = X.shape[0]
	k = w.shape[0]
	temp = X.dot(w.T)
	#return (((temp/np.column_stack((np.sum(temp, axis = 1),))).T - y.T).dot(X))/m
	return (1/np.sum(np.exp(np.reshape(temp, [m,k,1]) - np.reshape(temp, [m,1,k])), axis = 1) - y).T.dot(X)/m

def predict(X, w):
	return np.argmax(X.dot(w.T), axis = 1)

def getNewBatch(X, y, size):
	m = X.shape[0]
	d = X.shape[1]
	print X.shape, y.shape
	X_aug = np.hstack((X,y))
	np.random.shuffle(X_aug)
	start = np.random.randint(0,m-size)
	X_batch = X_aug[start:start+size,:d]
	y_batch = X_aug[start:start+size,d:]
	return X_batch, y_batch

def descent(X, y, w, alpha, stochastic = False, batch_size = 0):
	if stochastic:
		X, y = getNewBatch(X, y, batch_size)
	grad_w = evalGrad(X, y, w)
	return w - alpha*grad_w

def evaluate(X, y, w, classes):
	print "Class", "True Labels", "Predicted Labels", "Precision", "Recall" 
	pred = classes[predict(X,w)]
	y = classes[np.argmax(y)]
	true_counts = dict(zip(np.unique(y, return_counts = True)))
	pred_counts = dict(zip(np.unique(pred, return_counts = True)))
	pos_counts = dict(zip(np.unique(y[pred==y], return_counts = True)))
	for class_ex in classes:
		true = true_counts.get(class_ex, 0)
		predicted = pred_counts.get(class_ex, 0)
		pos = pos_counts.get(class_ex, 0)
		print class_ex, true, predicted, pos, float(pos)/predicted, float(pos)/true

def softmax(X, y, pad_ones=True, one_hot_encode=True, stochastic=False, batch_size=0):
	"""Return a softmax model and the corresponding test error. The training:testing divide used is 70:30"""
	m = X.shape[0] #Number of data points
	d = X.shape[1] #Number of features
	if pad_ones:
		X = np.pad(X, ((0,0),(1,0)), 'constant', constant_values=1)
		d += 1
	if one_hot_encode:
		classes = np.unique(y)
		classes_expanded = np.matlib.repmat(classes, m, 1)
		y = np.array(classes_expanded == np.column_stack((y,)), dtype = float)
	else:
		classes = np.arange(y.shape[1])
	k = y.shape[1]
	X_aug = np.hstack((X,y))
	np.random.shuffle(X_aug)
	m_train = m*7/10
	X_test = X_aug[m_train:,:d]
	y_test = X_aug[m_train:,d:]
	X = X_aug[:m_train,:d]
	y = X_aug[:m_train,d:]
	w = np.random.rand(k,d)*20 - np.ones((k,d))*10
	alpha = 2
	epsilon = 1e-5
	loss_new = evalLoss(X, y, w)
	loss_old = loss_new + 10**3
	while(abs(loss_old-loss_new) > epsilon):
		if stochastic:
			w = descent(X, y, w, alpha, stochastic=True, batch_size=batch_size)
		else:
			w = descent(X, y, w, alpha)
		loss_old = loss_new
		loss_new = evalLoss(X, y, w)
		print loss_old, loss_new
	evaluate(X, y, w, classes)
	return w, evalLoss(X_test, y_test, w)
