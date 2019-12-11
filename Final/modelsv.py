import numpy as np
import matplotlib.pyplot as plt
import glob
from unetparts import *
#### To approach this problem, i decided to bin the velocity data and classify using unet

#Rows in text files contain these in order
#velmag, x coord, y coord, ice thickness, bed elev, surface elev, routing
#0       1        2        3              4         5             6
### Data can be reshaped into 59 x 1085

## This bins the data
def continuetodiscrete(data, bins, shape):
        #data2 = (data-min(data))/(max(data)-min(data))*bins #rescale the data between zero and bin needed to scale down the max so there isn't one point in the highest point this if one data set
        data2 = data/442.8*bins  #min 0 max 442 for all of the data sets  
        return np.array([data2.astype(int).reshape(shape)])

files = glob.glob('*.txt')


### Split the images into traing and test data
X = []
X_test = []
y = []
y_test = []

numbins = 50

for i in range(len(files)): #to place images into test and training sets
        
        datas = np.genfromtxt(files[i], skip_header = 1,delimiter = ',')

        velmag = datas[0,:]
        data = datas[3:,:].T
        
        data2 = []
        
        for j in range(len(data[1,:])):
                data2.append(data[:,j].reshape(59,1085))
        data2 = np.array(data2)
        binvel = continuetodiscrete(velmag, numbins, (59,1085))
        if i>9:
                X.append(data2)
                y.append(binvel)
        else:
                X_test.append(data2)
                y_test.append(binvel)

X = np.array(X)
X_test = np.array(X_test)
y = np.array(y)
y_test = np.array(y_test)

### Define model ####
#################################################################
### need model that inputs 4 channels, and outputs velocity map
################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

plt.figure()
plt.plot(y_test[0,0,:,:], '.', alpha = .01)
plt.figure()
plt.imshow(y[0,0,:,:], interpolation = 'none')

print(X_test.shape, y_test.shape, X.shape, y.shape)


X = torch.from_numpy(X)
X_test = torch.from_numpy(X_test)
y = torch.from_numpy(y)
y_test = torch.from_numpy(y_test)

X = X.to(torch.float32)
X_test = X_test.to(torch.float32)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

X = X.to(device)
X_test = X_test.to(device)
y = y.to(device)
y_test = y_test.to(device)

from torch.utils.data import TensorDataset

training_data = TensorDataset(X,y)
test_data = TensorDataset(X_test,y_test)

batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                           batch_size=batch_size, 
                                           shuffle=True)

batch_size = 16
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size, 
                                           shuffle=True)


### UNET CNN model that imputs images and outputs an image from milesial/Pytorch-UNet on github ###

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

from torchsummary import summary

def train_model(epic):

        model = UNet(n_channels = 4, n_classes = numbins)
        model.to(device)

        summary(model,(4,100,1085)) #Get a model Sumarry  

        criterion = torch.nn.CrossEntropyLoss() #since ive set this up as a classification problem with bins number of classes

        optimizer = torch.optim.Adam(model.parameters())

        epochs = epic
        # Loop over the data

        for epoch in range(epochs):
                model.train()
                # Loop over each subset of data

                correct = 0
                total = 0        

                for d,t in train_loader:
                        # Zero out the optimizer's gradient buffer
                        optimizer.zero_grad()
                        # Make a prediction based on the model
                        outputs = model(d)
                        # Compute the loss
                        loss = criterion(outputs,t[:,0,:,:])
                        # Use backpropagation to compute the derivative of the loss with respect to the parameters
                        loss.backward()
                        # Use the derivative information to update the parameters
                        optimizer.step()

                        if epoch%100==0: #every once in a while see how the model is doing
                                _, predicted = torch.max(outputs.data, 1)
                                correct += len(predicted[predicted==t[:,0,:,:]])
                                total += len(predicted.flatten())
                
                if epoch%100==0:      
                        print(epoch,loss.item(), 'Accuracy = ', correct/total*100)

        return model


velmod = train_model(1000)
print('TRAINING COMPLETE')

#### A little post processing ####

interval = 442.8/numbins

for d, t in train_loader:        
    outputs = velmod(d)
    for i in range(10):
        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        correct += len(predicted[i][predicted[i]==t[i,0,:,:]])
        total += len(predicted[i].flatten())
        print(predicted.shape,'Accuracy = ', str(correct/total*100))
        plt.figure()
        plt.subplot(311)        
        plt.imshow(t[i,0,:,:]*interval, aspect = 'auto', interpolation = 'none', extent= (0,.150*1085,0,59*.150),vmin = 0, vmax = 200)
        plt.colorbar(label = 'Velocity (m/yr)')
        plt.ylabel('Distance (km)')
        plt.title('Target, Set Accuracy = '+ str(correct/total*100))
        plt.subplot(312)
        plt.colorbar(label = 'Velocity (m/yr)')
        plt.title('Guess')
        plt.ylabel('Distance (km)')
        plt.imshow(predicted[i]*interval,aspect = 'auto', interpolation = 'none',extent= (0,.150*1085,0,59*.150),vmin = 0, vmax = 200)
        plt.subplot(313)
        plt.colorbar(label = 'Velocity (m/yr)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Distance (km)')     
        plt.title('Difference Between Target and Guess')
        plt.imshow(abs(t[i,0,:,:]-predicted[i])*interval,aspect = 'auto', interpolation = 'none',extent= (0,.150*1085,0,59*.150),vmin = 0, vmax = 200)

for d, t in test_loader:        
    outputs = velmod(d)
    for i in range(10):
        correct = 0
        total = 0
        _, predicted = torch.max(outputs.data, 1)
        correct += len(predicted[i][predicted[i]==t[i,0,:,:]])
        total += len(predicted[i].flatten())
        print(predicted.shape,'Accuracy = ', str(correct/total*100))
        plt.figure()
        plt.subplot(311)        
        plt.imshow(t[i,0,:,:]*interval, aspect = 'auto', interpolation = 'none', extent= (0,.150*1085,0,59*.150),vmin = 0, vmax = 200)
        plt.colorbar(label = 'Velocity (m/yr)')
        plt.ylabel('Distance (km)')
        plt.title('Target, Set Accuracy = '+ str(correct/total*100))
        plt.subplot(312)
        plt.colorbar(label = 'Velocity (m/yr)')
        plt.title('Guess')
        plt.ylabel('Distance (km)')
        plt.imshow(predicted[i]*interval,aspect = 'auto', interpolation = 'none',extent= (0,.150*1085,0,59*.150),vmin = 0, vmax = 200)
        plt.subplot(313)
        plt.colorbar(label = 'Velocity (m/yr)')
        plt.ylabel('Distance (km)')     
        plt.title('Difference Between Target and Guess')
        plt.xlabel('Distance (km)')
        plt.imshow(abs(t[i,0,:,:]-predicted[i])*interval,aspect = 'auto', interpolation = 'none',extent= (0,.150*1085,0,59*.150),vmin = 0, vmax = 200)
for d, t in test_loader:        
    outputs = velmod(d)
    _, predicted = torch.max(outputs.data, 1)
    plt.figure()
    plt.hist(abs((t[:,0,:,:]-predicted[:]).flatten())*interval, alpha=1, bins = numbins)
    plt.xlabel('Binned Velocity Difference (m/yr)')
    plt.ylabel('Number in Bin')
    plt.title('Number of Guesses vs Difference in Velocity Between Guess and Actual')

plt.show()

