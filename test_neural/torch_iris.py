import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import pandas as pd
import numpy as np
iris = 'iris.data'
df=pd.read_csv('iris.data', sep=',')
data = df.values
names = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
input = np.array(data[:,0:4],dtype=float).reshape(len(data),4)
output = np.zeros(len(data),dtype=float).reshape(len(data),1)
for i in range(len(data)):
    output[i][0] = names[data[i][4]]

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,4)
        self.fc3 = nn.Linear(4,1)
        self.fc4 = nn.Linear(4,1)


    def forward(self,x):
        f = F.relu(self.fc1(x))
        f2 = F.relu(self.fc2(f))
        f3 = F.relu(self.fc3(f2).reshape(1,4))
        f4 = F.relu(self.fc4(f3))
        return f4


model = Net()

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
for epoch in range(100):
    for inp,out in zip(input,output):
        inp = torch.from_numpy(inp).reshape(4,1)
        out = torch.from_numpy(out).reshape(1,1)
        output = model(inp.float())
        loss = criterion(output,out.float())
        print("Epoch: %i\tLoss: %f" %(epoch,loss),end="\r")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("\n")
