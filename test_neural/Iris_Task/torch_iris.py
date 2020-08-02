import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(1,5)
        self.fc2 = nn.Linear(5,4)
        self.fc3 = nn.Linear(4,2)
        self.fc4 = nn.Linear(4,3)
        self.fc5 = nn.Linear(2,1)
        self.relu = nn.Sigmoid()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x.T).T)
        x = self.relu(self.fc5(x))
        return x.reshape(3,1)


def dataToTensor(x):
    data = torch.from_numpy(x)
    return data



filename = "./iris.data"
data = []
answer = []
with open(filename,"r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row)!=0:
            if row[4]=='Iris-setosa':
                ans=[1,0,0]
            elif row[4] ==  'Iris-versicolor':
                ans = [0,1,0]
            elif row[4] ==  'Iris-virginica':
                ans = [0,0,1]
            data.append(row[:-1])
            answer.append(ans)


d1 = data[:50]
d2 = data[50:100]
d3 = data[100:150]
a1 = answer[:50]
a2 = answer[50:100]
a3 = answer[100:150]
train = []
check = []
train_a = []
check_a = []
for n in range(50):
    if n<33:
        train.append(d1[n])
        train.append(d2[n])
        train.append(d3[n])
        train_a.append(a1[n])
        train_a.append(a2[n])
        train_a.append(a3[n])
    else:
        check.append(d1[n])
        check.append(d2[n])
        check.append(d3[n])
        check_a.append(a1[n])
        check_a.append(a2[n])
        check_a.append(a3[n])


train = np.array(train,dtype=np.float32)
train_a = np.array(train_a,dtype=np.float32)
check = np.array(check,dtype=np.float32)
check_a = np.array(check_a,dtype=np.float32)



model = Net()
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(20):
    for n in range(len(train)):
        data = dataToTensor(train[n]).reshape(4,1)
        ans = dataToTensor(train_a[n]).reshape(3,1)
        output = model(data)
#         print(output,"\n",ans,"\n","*****"*3)
#         print(output.size(),ans.size())
        loss = criterion(output,ans)

        print("Epoch: %i\tLoss: %f" %(epoch,loss),end="\r")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("\n")

correct = 0
total = len(check)
for n in range(total):
    checkList = []
    data = dataToTensor(check[n]).reshape(4,1)
    ans = dataToTensor(check_a[n]).float().reshape(1,3)
    output = model(data)
#     print(output)
#     print(ans)
    for i in output.data:

        if i.item() == torch.max(output.data):
            checkList.append(1)
        else:
            checkList.append(0)

    checkTensor = torch.tensor(checkList).reshape(1,3)
    if torch.all(torch.eq(checkTensor,ans)):
        correct+=1
    proc = (correct/total)*100
print("Total predict is: %f %%" %proc)
print("correct is: %f" %correct)
