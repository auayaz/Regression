import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(z):
    return 1/(1+np.exp(-z))



X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])

plt.figure(1)
plt.plot(X_train[:,0],X_train[:,1],'o')
plt.show()


def logistic_cost_function(X_train,y_train,w,b):

    J = 0
    m = X_train.shape[0]
    for i in range(m):
        z_tmp = np.dot(X_train[i],w) + b
        f_wb = sigmoid_function(z_tmp)
        J = J -y_train[i]*np.log(f_wb)-(1-y_train[i])*np.log(1- f_wb)
    J = (1/m)*J

    return J

def compute_gradient_logistic(X, y, w, b): 
    """
    computing the gradient
    """

    m,n = X.shape
    dj_dw = np.zeros((n,))                           
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid_function(np.dot(X[i],w) + b) 
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m                                   
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    performing gradient descent
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = w_in  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db
        tmp = logistic_cost_function(X,y,w,b)               
        J_history.append(tmp)

    return w, b, J_history         #return final w,b and J history for graphing





def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    m, n = X.shape   
    p = np.zeros(m)
   
    for i in range(m):
        z_wb = np.dot(X[i],w) + b
        #z_wb = sigmoid_function(z_tmp)
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij

        z_wb += b
        f_wb = sigmoid_function(z_wb)
        p[i] = f_wb >= 0.5
        
    return p

b_init = -3
w_init = np.array([1,1])
w_out, b_out, J_history = gradient_descent(X_train, y_train, w_init, b_init, alpha=0.1, num_iters=10000) 
print(w_out,b_out)

plt.plot(J_history,'.-')
plt.show()




"""
z = sigmoid_function(z_tmp)

plt.figure(1)
plt.plot(z_tmp,z,'-.r')
#plt.plot(z,'o-b')
plt.show()
"""
