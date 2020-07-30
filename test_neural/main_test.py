import numpy as np
import matplotlib.pyplot as plt
import theano
from theano import tensor as T
def sigm(data):
    x = T.dmatrix("x")
    s = 1/(1+T.exp(-x))
    sigm  = theano.function([x],s)
    return sigm(data)


input,output,w = T.dmatrices("input","output","w")
output = 1/(1+T.exp(-T.dot(w,input)))
# output = sigm(for_sigm)
f = theano.function([w,input],output)


input_data = np.array([1,0],float).reshape([2,1])
w1 = np.array(([0.45,-0.12,0.78,0.13]),float).reshape([2,2])
w2 = np.array(([1.5,-2.3]),float).reshape([1,2])

input_data2 = f(w1,input_data)
print(input_data2)
res = f(w2,input_data2)
print(res)
