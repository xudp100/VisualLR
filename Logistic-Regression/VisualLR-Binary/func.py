import numpy as np 

# Sigmoid function
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))


# 
def F(n, d, X, Y, w, lamb): 
	loss = 0
	for i in range(0,n): 
		w_Xi = np.dot(w,X[i,:])
		exp_YwX = np.exp(-Y[i]*w_Xi)
		if (exp_YwX > 1e20): 
			loss = loss + 20 / (n + 0.0) 
		else: 
			loss = loss + np.log(1 + exp_YwX) / (n + 0.0) 

	w_sq = np.square(w)
	reg = np.sum(np.divide(w_sq, 1 + w_sq))

	loss = loss.item() + (lamb/(2+0.0)) * reg		
	return loss


# Computing component gradient value
def grad_com(i, n, d, X, y, w, lamb):
	w_Xi = np.dot(w,X[i,:])
	exp_YwX = np.exp(-y[i]*w_Xi)
	if (exp_YwX > 1e20):
		grad = -y[i]*X[i,:]
	else: 
		grad = (-y[i]*X[i,:]*exp_YwX/(1 + exp_YwX))
	
	w_sq = np.square(w)
	grad_reg = np.divide(w, np.square(1 + w_sq))

	grad = grad + lamb * grad_reg

	return grad
	

# Computing batch gradient (indices)
def batch_grad(indices, n, d, X, Y, w, lamb):
	batch_grad = np.zeros(d)
	batchsize = len(indices)

	for i in indices:
		batch_grad = batch_grad + grad_com(i, n, d, X, Y, w, lamb)

	batch_grad = batch_grad/(batchsize+0.0)

	return batch_grad
    

# Computing accuracy
def accuracy(n, d, X, Y, w, lamb):
	acc = 0
	for i in range(0,n): 
		w_Xi = np.dot(w,X[i,:])
		if (Y[i]*w_Xi >= 0):
			acc = acc + 1
	acc_val = acc/(n+0.0)
	return acc_val
