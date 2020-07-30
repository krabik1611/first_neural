import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

data = torch.tensor([0,0,0,1,1,0,1,1],dtype=float).reshape(4,2)

out_data = torch.tensor([0,1,1,1],dtype=float).float().reshape(4,1)

# print(data.shape)
# for x,out in zip(data,out_data):
#     x = x.reshape(1,2)
#     print(x,out)
#     print(x.shape,out.shape)
#     print("\n")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(2,1)
        self.fc2 = nn.Linear(1,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net()

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)

for epoch in range(100):
    for x,out in zip(data,out_data):
        x = x.reshape(1,2)
        out = out.reshape(1,1)
        output = model(x.float())
        loss = criterion(output,out)
        # print(out.item())
        print("Epoch: %i\tLoss: %f\tOutput: %f" %(epoch,loss,output),end="\r")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("\n")

while 1:
    x1 = int(input())
    if x1==-1:
        break
    x2 = int(input())
    x = torch.tensor([x1,x2]).reshape(1,2).float()

    if model(x) > 0.5:
        print("Answer:1")
    else:
        print("Answer:0")
