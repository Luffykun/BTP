import numpy as np
import matplotlib.pyplot as plt

def gen_eigenvalues(variables, probs, num_classes, num_dims):
	variables = [1] + variables
	num_dims += 1
	variable_diag = np.diag([i for _ in range(num_classes) for i in variables])
	prob_diag = np.diag(probs)
	identity_rep = np.repeat(np.eye(num_classes), num_dims, axis = 1)
	prob_rep = np.reshape(np.repeat(probs, num_dims), (1,-1))
	A = identity_rep.T.dot(prob_diag.dot(identity_rep))
	B = prob_rep.T.dot(prob_rep)
	hessian = variable_diag.dot((A - B).dot(variable_diag))
	eigs = np.linalg.eigvalsh(hessian)
	return eigs

def plot_eigenvalue():
	prob_range = np.linspace(0,1,1001)
	pi_1, pi_2 = np.meshgrid(prob_range, prob_range)
	pi_3 = 1 - pi_1 - pi_2
	eigs_all = np.zeros([1001,1001,9])
	for i in range(1001):
		for j in range(1001):
			if pi_3[i,j] < 0:
				print i,j
				continue
			probs = [pi_1[i,j], pi_2[i,j], pi_3[i,j]]
			eigs_all[i,j,:] = gen_eigenvalues([1,1], probs, 3, 2)
	print np.max(eigs_all), np.min(eigs_all)
	fig, ax = plt.subplots()
	cs = ax.contourf(pi_1, pi_2, eigs_all[:,:,8])
	ax.set_aspect('equal')
	fig.colorbar(cs, ax=ax)
	plt.show()	

def plot_2D():
	pi_1 = np.linspace(0,1,1001)
	pi_2 = 1 - pi_1
	eigs_all = np.zeros([1001,6])
	for i in range(1001):
		probs = [pi_1[i], pi_2[i]]
		eigs_all[i,:] = gen_eigenvalues([1,1], probs, 2, 2)
	fig, axs = plt.subplots(3,2)
	for i, ax in enumerate(axs.flatten()):
		ax.plot(pi_1, eigs_all[:,i])
	plt.show()

if __name__ == '__main__':
	plot_eigenvalue()
	#plot_2D()
