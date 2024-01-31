import copy, math
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


def compute_cost(X, y, w, b):
    """
    This function computes the cost function J for the multivariable
    case.
    
    """
    n = X.shape[0]
    J = 0
    for i in range(n):
        f_wb = np.dot(X[i],w) + b
        J = J + (f_wb - y[i])**2

    J = (1/(2*n))*J

    return J


def compute_gradient(X, y, w, b):
    """
    This function computes the gradient of the cost function in the
    multivariable case.

    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m      

    return dj_db,dj_dw


def gradient_descent(X, y, w_in, b_in,alpha,iterations):
    """
    Gradient descent for multivariable linear regression
    """
    w = w_in
    b = b_in
    w_history = []; J_history = []; b_history = []
    i = 0

    while i<= iterations:
        
        dj_db,dj_dw = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J = compute_cost(X, y, w, b)

        w_history.append(w); b_history.append(b); J_history.append(J)

        i = i+1

    w_final = w
    b_final = b
    
    return w_history,b_history,w_final,b_final,J_history

initial_w = np.zeros_like(w_init)
initial_b = 0.
w_history,b_history,w_final,b_final,J_history = gradient_descent(X_train, y_train, initial_w, initial_b, 
                                                                 alpha = 5.0e-7,iterations = 1000)


def predict(x, w, b):
    p = np.dot(x,w) + b
    return p


#using appropriate dataset to train model.


"""
print(w_final,b_final)
print("the prediction")
p = predict(X_train[1,:],w_final,b_final)
#checking how well the model fits.
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")



plt.figure(2)
plt.plot(J_history[4:],'o-')
plt.title("cost function J")
plt.show()



def rescaling_features(x_train):

    #plotting features against the target variable 
    X = x_train
    plt.figure(1)
    fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    X_features = ['size(sqft)','bedrooms','floors','age']
    #Plotting for rescaling features.
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train)
        ax[i].set_xlabel(X_features[i])

    plt.show()

    #Rescaling the features with its maximum.
    for i in range(4):
        X[:,i] = x_train[:,i]/np.max(x_train[:,i])

    return X

X_train = rescaling_features(X_train)

#Running the gradient descent algorithm with rescaled values                   
w_history,b_history,w_final,b_final,J_history = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                                 alpha = 0.1 ,iterations = 50
)
print(w_final,b_final)

plt.figure(3)
plt.plot(J_history[0:],'o-')
plt.title("cost function J")
plt.show()
"""

