import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(4,10),
            nn.Sigmoid(),
            nn.Linear(10,3),
            nn.Sigmoid()
        )


    def forward(self,x):
        return self.layer1(x).view(-1)


def data2tensor(data):
    for n in range(len(data)):
        data[n] = float(data[n])

    return torch.Tensor(data).reshape([1,4]).float()

def loadData():
    def convert(*data):
        values = []
        for n in data:
            if n[-1] == "Iris-setosa":
                ans = [1,0,0]
            elif n[-1] == "Iris-versicolor":
                ans = [0,1,0]
            elif n[-1] == "Iris-virginica":
                ans = [0,0,1]
            values.append([n[:-1],ans])

        return values


    filename = "iris.data"
    data = []
    test= []
    count=count_= 0
    with open(filename, "r") as f:
        reader = list(csv.reader(f))[:-1]
        setosa,versicolor,virginica = [reader[x:x+50] for x in range(0,150,50)]

    for d1,d2,d3 in zip(setosa,versicolor,virginica):
        if count <37:
            for n in convert(d1,d2,d3):
                data.append(n)
            count +=1
        else:
            for n in convert(d1,d2,d3):
                test.append(n)

    return (data,test)


def learn():
    net = Net()
    # net.load_state_dict(torch.load("model.th"))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.0005)
    train,test = loadData()
    for n in range(100):
        for i,(data,label) in enumerate(train):
            data = data2tensor(data)
            label = torch.Tensor(label).float()
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
            print("Epoch: {}\t Count: {}\tLoss: {}".format(n,i,loss.item()))

    torch.save(net.state_dict(),"model.th")


def run(choose):
    net = Net()
    net.load_state_dict(torch.load("model.th"))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.0005)
    train,test = loadData()
    correct = 0
    if choose ==1:
        dataset = train
    elif choose == 2:
        dataset = test
    with torch.no_grad():
        for data,label in dataset:
            data = data2tensor(data)
            label = torch.Tensor(label).float()
            outputs = net(data)
            if torch.max(label,0)[1] == torch.max(outputs,0)[1]:
                correct +=1
        print(correct*100/len(dataset))


def main():
    while 1:
        ans = int(input("1)Обучить\n2)Протестировать\n3)Выйти\nОтвет:>"))
        if ans == 1:
            learn()
        elif ans == 2:
            ans = int(input("Запустить на тренировочных или тестовых данных?\n1)Тренирочные\n2)тестовыe\nОтвет:>"))
            if ans == 1 :
                run(1)
            elif ans ==2:
                run(2)
        elif ans ==3:
            break

if __name__ == '__main__':
    main()
