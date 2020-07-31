import torch
import torch.nn as nn
import requests
import pandas as pd
import numpy as np
iris = 'iris.data'
df=pd.read_csv('iris.data', sep=',')
data = df.values
names = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
input = np.array(data[:,0:3])
output = np.array(data[:,4])
for i in range(len(output)):
    output[i] = names[output[i]]

class Net(nn.Module):
    def __init__(self):
        supet(Net,self).__init__()
        self.fc1 = nn.Liear(1,4)
        
print(len(output))
for inp,out in zip(input,output):
    print(inp,out)
