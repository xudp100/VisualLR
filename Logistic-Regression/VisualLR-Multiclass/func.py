import numpy as np 

"""
n: number of the samples 
d: dimension of the feature
n_classes: number of the classes 
X: (n x d)
Y: one-hot (n x n_classes)
W: (d x n_classes)
lamb: regularization coefficient
"""
    
# Softmax function for multi-class classification
def softmax(z):
    
    if z.ndim == 1:
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)  
    else:
        z = z - np.max(z, axis=1, keepdims=True)  
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  
    


# Multi-class logistic regression loss function
def F_multi_class(n, d, n_classes, X, Y, W, lamb):

    loss = 0
    
    # 
    scores = X @ W  # (n x n_classes)
    probs = softmax(scores)  # (n x n_classes)
    
    loss = 0
    for i in range(n):  
        for k in range(n_classes): 
            # Y[i,k] * log(probs[i,k])
            loss += Y[i, k] * np.log(probs[i, k] + 1e-15)  
            
    loss = -loss / n
    
    # 
    w_sq = np.square(W)
    reg = np.sum(np.divide(w_sq, 1 + w_sq))    
    
    loss = loss + (lamb / 2.0) * reg
    return loss


# Computing component gradient value for multi-class
def grad_com_multi(i, n, d, n_classes, X, Y, W, lamb):

    xi = X[i, :].flatten()  
    yi = Y[i, :].flatten()  
    
    scores_i = xi @ W  # (n_classes,)
    probs_i = softmax(scores_i)  # (n_classes,)
    
    error = probs_i - yi  # (n_classes,)
    grad = xi.reshape(-1, 1) @ error.reshape(1, -1)  # (d, n_classes)

    w_sq = np.square(W)
    grad_reg = np.divide(W, np.square(1 + w_sq))
    
    return grad + lamb * grad_reg


# Computing batch gradient for multi-class
def batch_grad_multi(indices, n, d, n_classes, X, Y, W, lamb):

    batch_grad = np.zeros((d, n_classes))
    batchsize = len(indices)
    
    
    for idx in indices:
        grad_i = grad_com_multi(idx, n, d, n_classes, X, Y, W, lamb)
        batch_grad += grad_i
    
    batch_grad = batch_grad / batchsize
    return batch_grad

# Computing accuracy for multi-class
def accuracy_multi_class(n, d, n_classes, X, Y, W, lamb):
   
    acc = 0
    scores = X @ W  # (n x n_classes)
    predictions = np.argmax(scores, axis=1)  
    true_labels = np.argmax(Y, axis=1)  
    
    for i in range(n):
        if predictions[i] == true_labels[i]:
            acc += 1
    
    acc_val = acc / n
    return acc_val
