"""
@author: Allan Santos
"""

import numpy as np

"""for linear separable problems (only ml)"""
def stepFunction(soma):
    if( soma >= 1):
        return 1
    else: 
        return 0
    
""" used for two classes problems (binary classfication)"""
def sigmoidFunction(soma):
    return 1 / (1  + np.exp(-soma))
    

""" used for classification"""
def hyperbolicTanh(soma):
    return (np.exp(soma) - np.exp(-soma))  / ((np.exp(soma) + np.exp(-soma)))


""" used a lot for CNN or with neural nets with many layers """
def reluFunction(soma):
    if(soma < 0):
        return 0
    else:
        return soma
"""used for regression problems: just return the input value """
def linearFunction(soma):
    return soma

"""used for probabilities for more than 2 labels/classes/y"""
def softmaxFunction(x_vector):
    ex = np.exp(x_vector)
    
    return ex / ex.sum()

    
print("step:", stepFunction(-1))

print("sigmoid:", sigmoidFunction(-0.577))

print("hyperbolic:", hyperbolicTanh(-0.358))

print("relu:", reluFunction(-1))

print("relu:", reluFunction(10))

print("linear:", linearFunction(10))

values_vector = [5.0, 2.0, 1.3]

print("softmax:", softmaxFunction(values_vector))
