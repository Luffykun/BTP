import numpy as np
import numpy.matlib
import scipy
from scipy.optimize import minimize

def pad(X):
	return np.pad(X, ((0,0),(1,0)), 'constant', constant_values=1)

def encode(y, classes):
	classes_expanded = np.matlib.repmat(classes, y.shape[0], 1)
	return np.array(classes_expanded == np.column_stack((y,)), dtype = int)

def pad_encode(X, y, classes, to_pad=True, to_encode=True):
	X_new = pad(X) if to_pad else X
	y_new = encode(y, classes) if to_encode else y
	return X_new, y_new

def retdims(X, y):
	m = X.shape[0]
	d = X.shape[1]
	if len(y.shape) == 2:
		k = y.shape[1]
	else:
		k = 2
	return m, d, k

def division_permutations(num, ratio = 0.7, seed = 5):
	np.random.seed(seed = seed)
	indices = np.arange(0, num, 1)
	np.random.shuffle(indices)
	train_num = int(num*ratio)
	train_permutation = indices[:train_num]
	test_permutation = indices[train_num:]
	return train_permutation, test_permutation

def divide_dataset(X, y, train_permutation, test_permutation):
	X_test = X[test_permutation]
	y_test = y[test_permutation]
	X_train = X[train_permutation]
	y_train = y[train_permutation]
	return X_train, y_train, X_test, y_test

def getNewBatch(X, y, size):
	m, d, k = retdims(X, y)
	X_dtype = X.dtype
	y_dtype = y.dtype
	X = np.array(X, dtype=object)
	dim_changed = False
	if y.ndim == 1:
		y = np.reshape(y, (-1,1))
		dim_changed = True
	y = np.array(y, dtype=object)
	X_aug = np.hstack((X,y))
	np.random.shuffle(X_aug)
	start = np.random.randint(0,m-size)
	X_batch = np.array(X_aug[start:start+size,:d], dtype=X_dtype)
	y_batch = np.array(X_aug[start:start+size,d:], dtype=y_dtype)
	if dim_changed:
		y_batch = np.reshape(y_batch, (-1,))
	return X_batch, y_batch

def secant(fun, max_iter=1000, epsilon=1e-2):
	x1 = 0
	x2 = 2
	val1 = fun(x1)
	val2 = fun(x2)
	while val1*val2 > 0:
		x2 +=2
		val2 = fun(x2)
	i = 0
	while abs(val2) > epsilon and i < max_iter:
		temp = x2
		if val1 == val2:
			import pdb; pdb.set_trace()
			print val1, val2, x1, x2
		x2 = x2 - val2*(x2 - x1)/(val2 - val1)
		x1 = temp
		val1 = val2
		val2 = fun(x2)
	return x2

def descent(X, y, w, grad_fn, stochastic = False, batch_size = 0):
	if stochastic:
		X, y = getNewBatch(X, y, batch_size)
		#alpha = 2
		grad_w = grad_fn(X, y, w)
		alpha = secant(lambda alpha: -1*np.sum(grad_fn(X, y, w - alpha*grad_w)*grad_w), 3)
		return w - alpha*grad_fn(X, y, w), alpha
	else:
		grad_w = grad_fn(X, y, w)
		#alpha = 2
		alpha = secant(lambda alpha: -1*np.sum(grad_fn(X, y, w - alpha*grad_w)*grad_w), 3)
		return w - alpha*grad_w, alpha

def gradient_descent(X, y, w0, loss_fn, grad_fn, threshold = 1e-5, stochastic=False, batch_size=0):
	w = w0
	#loss_new = loss_fn(X, y, w)
	#loss_old = loss_new + 1e3
	grad_norm = np.linalg.norm(grad_fn(X, y, w))
	alpha = 1
	iteration_count = 0
	while(grad_norm > threshold):
		iteration_count += 1
		if stochastic:
			for _ in range(10):
				w, alpha = descent(X, y, w, grad_fn, stochastic=True, batch_size=batch_size)
		else:
			w, alpha = descent(X, y, w, grad_fn)
		#loss_old = loss_new
		#loss_new = loss_fn(X, y, w)
		grad_norm = np.linalg.norm(grad_fn(X, y, w))
		#print iteration_count, grad_norm
		#print iteration_count, loss_old, loss_new
	return w

def evaluate(X, y, w, classes, predict_fn):
	pred = predict_fn(X, w, classes)
	true_counts = dict(zip(*np.unique(y, return_counts = True)))
	pred_counts = dict(zip(*np.unique(pred, return_counts = True)))
	pos_counts = dict(zip(*np.unique(y[pred==y], return_counts = True)))
	precisions = {}
	recalls = {}
	print "Class", "True Labels", "Predicted Labels", "True Positives", "Precision", "Recall", "F1-Score"
	for class_ex in classes:
		true = true_counts.get(class_ex, 0)
		predicted = pred_counts.get(class_ex, 0)
		pos = pos_counts.get(class_ex, 0)
		precision = float(pos)/predicted if predicted > 0 else 0
		precisions[class_ex] = precision
		recall = float(pos)/true if predicted > 0 else 0
		recalls[class_ex] = recall
		f1_score = 2*precision*recall/(precision + recall) if (precision + recall > 0) else 0
		print "%s %d %d %d %.2f %.2f %.2f" % (class_ex, true, predicted, pos, precision, recall, f1_score)
	print encode(pred, classes).T.dot(encode(y, classes))
	sum_true = np.sum(true_counts.values())
	sum_predicted = np.sum(pred_counts.values())
	sum_pos = np.sum(pos_counts.values())
	micro_precision = float(sum_pos)/sum_predicted
	micro_recall = float(sum_pos)/sum_true
	micro_f1 = 2*micro_precision*micro_recall/(micro_precision + micro_recall) if (micro_precision + micro_recall > 0) else 0
	macro_precision = np.mean(precisions.values())
	macro_recall = np.mean(recalls.values())
	macro_f1 = 2*macro_precision*macro_recall/(macro_precision + macro_recall) if (macro_precision + macro_recall > 0) else 0
	print "Micro Precision:", micro_precision
	print "Micro Recall:", micro_recall
	print "Micro F1 Score", micro_f1
	print "Macro Precision:", macro_precision
	print "Macro Recall:", macro_recall
	print "Macro F1 Score", macro_f1

def evalLoss(X, y, w):
	m, d, k = retdims(X, y)
	temp = X.dot(w.T) - np.sum(X*(y.dot(w)), axis = 1, keepdims = True)
	temp2 = np.max(temp, axis = 1, keepdims = True)
	return (np.sum(np.log(np.sum(np.exp(temp - temp2), axis = 1)) + np.reshape(temp2, [-1])))/m

def evalGrad(X, y, w):
	m, d, k = retdims(X, y)
	temp = X.dot(w.T)
	return (1/np.sum(np.exp(np.reshape(temp, [m,k,1]) - np.reshape(temp, [m,1,k])), axis = 1) - y).T.dot(X)/m

def predict(X, w, classes):
	X = pad(X)
	return classes[np.argmax(X.dot(w.T), axis = 1)]

def softmax_preprocess(X, y, pad_ones, one_hot_encode, seed):
	if one_hot_encode:
		classes = np.unique(y)
	else:
		classes = np.arange(y.shape[1])
	train_permutation, test_permutation = division_permutations(X.shape[0], seed = seed)
	X, y, X_test, y_test = divide_dataset(X, y, train_permutation, test_permutation)
	X_pad, y_enc = pad_encode(X, y, classes, pad_ones, one_hot_encode)
	return X_pad, y_enc, X_test, y_test, classes

def softmax(X, y, pad_ones = True, one_hot_encode = True, threshold = 1e-5, stochastic = False, batch_size = 0, seed = 5):
	"""Return a softmax model and the corresponding test error. The training:testing divide used is 70:30"""
	X_pad, y_enc, X_test, y_test, classes = softmax_preprocess(X, y, pad_ones, one_hot_encode, seed)
	m, d, k = retdims(X_pad, y_enc)
	w = np.random.rand(k,d)*20 - np.ones((k,d))*10
	w = gradient_descent(X_pad, y_enc, w, evalLoss, evalGrad,threshold=threshold, stochastic=stochastic, batch_size=batch_size)
	evaluate(X_test, y_test, w, classes, predict)
	result = {}
	result["model"] = w
	result["test_error"] = evalLoss(pad(X_test), encode(y_test, classes), w)
	print "Error", result["test_error"]
	return result

def softmax_bfgs(X, y, pad_ones = True, one_hot_encode = True, seed = 5, threshold = 1e-5):
	X_pad, y_enc, X_test, y_test, classes = softmax_preprocess(X, y, pad_ones, one_hot_encode, seed)
	m, d, k = retdims(X_pad, y_enc)
	w0 = (np.random.rand(k,d)*20 - np.ones((k,d))*10).flatten()
	model =  minimize(lambda w: evalLoss(X_pad, y_enc, np.reshape(w, (k,d))), w0, method = 'BFGS', jac = lambda w: evalGrad(X_pad, y_enc, np.reshape(w, (k,d))).flatten(), options = {'gtol': threshold}).x
	model = np.reshape(model, (k,d))
	evaluate(X_test, y_test, model, classes, predict)
	result = {}
	result["model"] = model
	result["test_error"] = evalLoss(pad(X_test), encode(y_test, classes), model)
	print "Error", result["test_error"]
	return result

def sig(A):
	return 1/(1 + np.exp(-A))

def logisticLoss(X, y, w, test = False):
	w = np.reshape(w, (-1,1))
	y = np.reshape(y, (-1,1))
	m = X.shape[0]
	temp1 = X.dot(w)
	pos = temp1[y==1]
	neg = temp1[y==0]
	sum1 = np.sum(np.log(1 + np.exp(-pos[pos > 0])))
	sum2 = np.sum(-pos[pos <= 0] + np.log(1 + np.exp(pos[pos <= 0])))
	sum3 = np.sum(np.log(1 + np.exp(neg[neg <= 0])))
	sum4 = np.sum(neg[neg > 0] + np.log(1 + np.exp(-neg[neg > 0])))
	return sum1 + sum2 + sum3 + sum4

def logisticGrad(X, y, w):
	w = np.reshape(w, (-1,1))
	y = np.reshape(y, (-1,1))
	m = X.shape[0]
	return -np.reshape(X.T.dot(y - sig(X.dot(w)))/m, (-1,))

def logistic(X, y, stochastic=False, batch_size=0):
	d = X.shape[1]
	y = np.reshape(y, (-1,1))
	w = np.random.rand(d,)*20 - np.ones((d,))*10
	w = gradient_descent(X, y, w, logisticLoss, logisticGrad, threshold=1e-3, stochastic=stochastic, batch_size=batch_size)
	return w

def ovo_predict(X, w, classes):
	X = pad(X)
	k = w.shape[0]
	assert w.shape[1] == k
	assert w.shape[2] == X.shape[1]
	m = X.shape[0]
	preds = np.array(np.zeros((m,)), dtype = int)
	for p in range(m):
		highest_bid = dict(zip([i for i in range(k)],[0 for _ in range(k)]))
		for q in range(k):
			for r in range(q+1,k):
				dot = X[p].dot(np.reshape(w[q,r], (-1,1)))
				if dot < 0:
					highest_bid[r] += 1
				elif dot > 0:
					highest_bid[q] += 1
		preds[p] = max(highest_bid, key=highest_bid.get)
	return classes[preds]

def binary_extension_preprocess(X, y, pad_ones, seed):
	classes = np.unique(y)
	train_permutation, test_permutation = division_permutations(X.shape[0], seed = seed)	
	X, y, X_test, y_test = divide_dataset(X, y, train_permutation, test_permutation)
	if pad_ones:
		X = pad(X)
	return X, y, X_test, y_test, classes

def one_vs_one(X, y, pad_ones = True, stochastic = False, batch_size = 0, seed = 5):
	X, y, X_test, y_test, classes = binary_extension_preprocess(X, y, pad_ones, seed)
	m = X.shape[0] #Number of data points
	d = X.shape[1] #Number of features
	k = classes.size
	X_exploded = []
	for a_class in classes:
		assert len(X[np.where(y==a_class)[0]].shape) == 2
		assert X[np.where(y==a_class)[0]].shape[1] == X.shape[1]
		X_exploded.append(X[np.where(y==a_class)[0]])
	w = np.zeros((k,k,d))
	for i in range(len(classes)):
		for j in range(i+1, len(classes)):
			y = [1 for _ in range(len(X_exploded[i]))]
			y += [0 for _ in range(len(X_exploded[j]))]
			log = logistic(np.vstack((X_exploded[i], X_exploded[j])), np.array(y), stochastic=stochastic, batch_size=batch_size)
			w[i,j] = np.copy(log)
			w[j,i] = -1*np.copy(log)
	evaluate(X_test, y_test, w, classes, ovo_predict)
	return w

def ova_predict(X, w, classes):
	X = pad(X)
	return classes[np.argmax(X.dot(w.T), axis = 1)]

def one_vs_all(X, y, pad_ones = True, stochastic = False, batch_size = 0, seed = 5):
	X, y, X_test, y_test, classes = binary_extension_preprocess(X, y, pad_ones, seed)
	m = X.shape[0] #Number of data points
	d = X.shape[1] #Number of features
	k = classes.size
	w = np.zeros((k,d))
	for i in range(len(classes)):
		y_bin = np.array(y == classes[i])
		w[i] = logistic(X, y_bin, stochastic = stochastic, batch_size = batch_size)
	evaluate(X_test, y_test, w, classes, ova_predict)
	return w

def svm_preprocess(X, y, pad_ones, one_hot_encode, seed):
	if one_hot_encode:
		classes = np.unique(y)
	else:
		classes = np.arange(y.shape[1])
	train_permutation, test_permutation = division_permutations(X.shape[0], seed = seed)	
	X, y, X_test, y_test = divide_dataset(X, y, train_permutation, test_permutation)
	X_pad, y_enc = pad_encode(X, y, classes)
	return X_pad, y_enc, X_test, y_test, classes

def svm_loss(X, y, w, delta):
	margin_loss = np.sum(np.max(y.dot(delta) + X.dot(w.T) - np.sum(X*(y.dot(w)), axis = 1, keepdims = True), axis = 1))
	return np.linalg.norm(w, 'fro') + margin_loss

def train_svm(X, y, delta=None, pad_ones=True, one_hot_encode=True, seed = 5):
	X_pad, y_enc, X_test, y_test, classes = svm_preprocess(X, y, pad_ones, one_hot_encode, seed)
	m, d, k = retdims(X_pad, y_enc)
	w0 = (np.random.rand(k,d)*20 - np.ones((k,d))*10).flatten()
	if delta is None:
		delta = np.ones((k,k)) - np.eye(k)
	model = minimize(lambda w: svm_loss(X_pad, y_enc, np.reshape(w, (k,d)), delta), w0).x
	model = np.reshape(model, (k,d))
	evaluate(X_test, y_test, model, classes, predict)
	result = {}
	result["model"] = model
	result["test_error"] = svm_loss(pad(X_test), encode(y_test, classes), model, delta)
	print "Error", result["test_error"]
	return result
