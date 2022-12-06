import numpy as np

def g_x(func_type, z):
    if func_type == 'softmax':
        z = np.exp(z-np.max(z)) / np.sum(np.exp(z-np.max(z)), axis=0, keepdims=True)
    elif func_type == 'sigmoid':
        z = 1/(1+np.exp(-z))

    elif func_type == 'relu':
        z[z<0] = 0

    elif func_type == 'tanh':
        z = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        
    return z

def gPrime_x(func_type, z):
    if func_type == 'softmax':
        zPrime = (z @ np.ones((1, len(z)))) * (np.identity(len(z)) - (np.ones((len(z), 1))@ z.T))

    elif func_type == 'sigmoid':
        zPrime = (1/(1+np.exp(-z)))*(1-(1/(1+np.exp(-z))))

    elif func_type == 'relu':
        zPrime = np.where(z>0,1,0)
        
    elif func_type == 'tanh':
        zPrime = (1 - np.power(z, 2))
    
    return zPrime
            