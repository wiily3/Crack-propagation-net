import torch
import scipy.io as sio
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class LoadData(object):
    def __init__(self):
        self.image_shape = sio.loadmat('./Arrays2/Sample1.mat')['Ux']
        self.width, self.height = self.image_shape.shape[0], self.image_shape.shape[1]
        self.timesteps = self.image_shape.shape[2]
        self.channels = 1

        for samples in range(450):
            data = sio.loadmat('./Arrays2/Sample'+str(samples+1)+'.mat')

            Crack = data['Crack'][..., 0]
            damage = data['Crack'].transpose() - Crack

            # add aditional dimision of channels
            Crack = np.expand_dims(Crack, 0)
            damage = np.expand_dims(damage, 0)

            # add aditional dimision of samples
            Crack = np.expand_dims(Crack, 0)
            damage = np.expand_dims(damage, 0)

            if samples == 0:
                self.microstruc = Crack
                self.damage = damage
            else:
                self.microstruc = np.concatenate((self.microstruc, Crack), axis=0)
                self.damage = np.concatenate((self.damage, damage), axis=0)

        global input_channel_num, output_channel_num
        input_channel_num = self.microstruc.shape[1] + self.damage.shape[1] # exclude last channel of mask
        output_channel_num = self.damage.shape[1] # channels number of one hot data

        

    def dataloader(self, batch_size):
        X_train, X_test, Y_train, Y_test = train_test_split(self.microstruc, self.damage, test_size=0.2, shuffle=True)
        X_train, X_test = torch.Tensor(X_train).float(), torch.Tensor(X_test).float()
        Y_train, Y_test = torch.tensor(Y_train).float(), torch.tensor(Y_test).float() 


        trainloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=True, num_workers=0)
        return trainloader, testloader


class Crack_Growth(nn.Module):
    def __init__(self, nc):
        super(Crack_Growth, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel_num, nc, (5, 5), padding=(2, 2)),  # dimension (50, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d((2,2)), # dimension (50, 64, 64)
            nn.Conv2d(nc, nc*2, (5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), # dimension (50, 32, 32)
            nn.Conv2d(nc*2, nc*4, (5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), # dimension (50, 16, 16)
        )

        self.cnn_t = nn.Sequential(
            nn.ConvTranspose2d(nc*4, nc*2, (4, 4), stride=(2, 2), padding=(1, 1)),  # dimension (50, 32, 32)]
            nn.ReLU(),
            nn.ConvTranspose2d(nc*2, nc*2, (4, 4), stride=(2, 2), padding=(1, 1)),  # dimension (50, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(nc*2, output_channel_num, (4, 4), stride=(2, 2), padding=(1, 1)),  # dimension (50, 128, 128)
            nn.ReLU()
        )

    def mask_func(self, x):

        mask = x
        return mask

    def forward(self, x):
        pre = self.cnn(x)
        pre = self.cnn_t(pre)
        mask = self.mask_func(x[:, 1:, ...])
        pre = torch.mul(pre, mask)
        return pre


class Loss(nn.Module):
    def __init__(self, pred, target):
        super(Loss, self).__init__()
        self.pred = pred
        self.target = target


    def bceloss(self):
        loss = nn.functional.cross_entropy(self.pred, self.target, weight=torch.Tensor([1,50]).cuda(),reduce=False)
        loss = torch.mean(torch.mul(loss, self.mask))
        return loss

    def mseloss(self):
        loss = nn.functional.mse_loss(self.pred, self.target)
        return loss


if __name__ == '__main__':
    epochs = 100
    batch_size = 10
    nc = 2

    trainloader, testloader = LoadData().dataloader(batch_size)

    net = Crack_Growth(nc)
    print(net)
    if torch.cuda.is_available():
        GPU = 1
        print('Training on GPU')
        net = net.cuda()
    print(net)

    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.001)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)


    # loss_func = nn.CrossEntropyLoss(reduce=False)

    trainloss, testloss = [], []

    for epoch in range(epochs):
        training_loss = 0
        it_num = 0

        for X_train, Y_train in trainloader:

            it_num += 1

            if GPU:
                X_train, Y_train = Variable(X_train).cuda(), Variable(Y_train).cuda()                
                
            optimizer.zero_grad()


            damage_accumalation = torch.zeros(X_train.shape).cuda() # X_train dimension: minibatch, channels, width, height
            # mask = torch.unsqueeze(torch.ones(X_train.shape), 1)
            for timesteps in range(Y_train.shape[2]):
                damage_increment = net(torch.cat((damage_accumalation, X_train), 1))
                damage_accumalation += damage_increment
                damage_accumalation = torch.clamp(damage_accumalation, min=0, max=1) #constrain damage in range(0,1)
                # mask.where(damage_accumalation > 1, torch.tensor(0).cuda)
                if timesteps == 0:
                    damage_tensor = torch.unsqueeze(damage_accumalation, 2)
                    # mask_tensor = mask
                else:
                    damage_tensor = torch.cat((damage_tensor, torch.unsqueeze(damage_accumalation, 2)), 2)
                    # mask_tensor = torch.cat((mask_tensor, mask), 2)

            

            loss = Loss(damage_tensor, Y_train).mseloss()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss = training_loss / it_num

        # if epoch % 100 == 0:
        #     print('Epoch: {}/{} \t Mean Square Error Loss: {}'.format(epoch + 1, epochs, training_loss))

        with torch.no_grad():
            # test dataset
            X_test, Y_test = next(iter(testloader))

            if GPU:
                X_test, Y_test = Variable(X_test).cuda(), Variable(Y_test).cuda()

            damage_accumalation = torch.zeros(X_train.shape).cuda() # X_train dimension: minibatch, channels, width, height
            # mask = torch.unsqueeze(torch.ones(X_train.shape), 1)
            for timesteps in range(Y_train.shape[2]):
                damage_increment = net(torch.cat((damage_accumalation, X_train), 1))
                damage_accumalation += damage_increment
                damage_accumalation = torch.clamp(damage_accumalation, min=0, max=1) #constrain damage in range(0,1)
                # mask.where(damage_accumalation > 1, torch.tensor(0).cuda)
                if timesteps == 0:
                    damage_tensor = torch.unsqueeze(damage_accumalation, 2)
                    # mask_tensor = mask
                else:
                    damage_tensor = torch.cat((damage_tensor, torch.unsqueeze(damage_accumalation, 2)), 2)
                    # mask_tensor = torch.cat((mask_tensor, mask), 2)
            testing_loss = Loss(damage_tensor, Y_train,).mseloss()

        trainloss.append(training_loss)
        testloss.append(testing_loss)


        # X_pred = torch.argmax(nn.functional.softmax(X_pred, dim=1), dim=1, keepdim=True)  # revert one hot   
        # print('maximum value of prediction data:' ,torch.max(X_pred))

        print('Epoch: {}/{} \t TrainLoss: {} \t TestLoss: {}'.format(epoch + 1, epochs, training_loss, testing_loss))

        torch.save(net, 'torch_model.pkl')



