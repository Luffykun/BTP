import numpy as np
import matplotlib.pyplot as plt

def gen_eigenvalues(variables, probs, num_classes, num_dims):
	variables = [1] + variables
	num_dims += 1
	variable_diag = np.diag([i for _ in range(num_classes) for i in variables])
	prob_diag = np.diag(probs)
	identity_rep = np.repeat(np.eye(num_classes), num_dims, axis = 1)
	prob_rep = np.repeat(probs, num_dims)
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
			probs = [pi_1[i,j], pi_2[i,j], pi_3[i,j]]
			eigs_all[i,j,:] = gen_eigenvalues([1,1], probs, 3, 2)
	print eigs_all
	fig, axs = plt.subplots(3,3)
	for i, ax in enumerate(axs.flatten()):
		levels = np.linspace(-50,3,54)
		if i == 0:
			cs = ax.contourf(pi_1, pi_2, eigs_all[:,:,8], levels)
			
			print levels
			print cs.vmin
		else:
			ax.contourf(pi_1, pi_2, eigs_all[:,:,8-i], levels)
		ax.set_aspect('equal')
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
	fig.colorbar(cs, cax=cbar_ax)
	plt.show()	

if __name__ == '__main__':
	plot_eigenvalue()
