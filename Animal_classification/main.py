import torchvision.datasets as dset
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 2704)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])


def show(tensor,out):
    global classes
    global count
    # print(tensor)
    no_batch = tensor.squeeze(dim=0)

    width_height_channels = no_batch.permute(1, 2, 0)
    # print(width_height_channels)
    # img = width_height_channels.mul(255)

    final_image = width_height_channels.numpy()
    # print(img)
    plt.subplot(1,5,count),plt.imshow(final_image)
    plt.title(classes[torch.max(out,1)[1].item()]),plt.xticks([]),plt.yticks([])




def getAns(x):
    ans = torch.zeros((1,10))
    ans[:,x.item()]=1
    return ans
trainset = dset.ImageFolder(root="archive/raw-img",transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,shuffle=True,batch_size=1)


classes = ("Dog","Horse","Enephante","Butterfly","Chiken","Cat","Cow","Sheep","Spider","Squirrel")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def learn():
    global net
    global trainloader
    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)


    all_loss=[]
    for epoch in range(10):
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs, labels = data
            labels = getAns(labels).to((device))
            # print(labels.size())

            optimizer.zero_grad()

            outputs = net(inputs.to(device))
            # print(outputs.size())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            if i%2000==1999:
                all_loss.append(running_loss)
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(net.state_dict(),"model.th")
    plt.plot(all_loss)
    plt.show()
    print("Finish Traning")
def showImg(array):
    for i,img in enumerate(array,1):
        plt.subplot(2,8,i),plt.imshow(img)
        plt.title(i),plt.xticks([]),plt.yticks([])
    plt.show()


def test():
    global net
    global trainloader
    global count
    count = 1
    net = Net()
    # net.load_state_dict(torch.load("model.th"))

    try:
        with torch.no_grad():
            for data in trainloader:
                # if count < 6:
                #     inputs, labels = data
                #
                #     outputs = net(inputs)#.to(device))
                #     show(inputs,outputs)
                #     count +=1
                # else:
                #     plt.show()
                #     count = 1
                inputs, labels = data
                output1 = net(inputs)
                net.load_state_dict(torch.load("model.th"))
                output2 = net(inputs)
                showImg(output1[0].detach().numpy())
                showImg(output2[0].detach().numpy())

                break
    except  KeyboardInterrupt:
        return 0
                                # plt.show()
# net = Net()#.to(device)
# learn()
test()
