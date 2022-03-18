# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:51:57 2020

@author: Avery Stehr
"""



import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Net(torch.nn.Module): # Cutsom ANN inherits from base class
    def __init__(self): # Default constuctor
        super(Net,self).__init__() # Run base class const
        self.fc1 = torch.nn.Linear(2,1) # Two in one out
        self.sigmoid = torch.nn.Sigmoid() #Sigmoid for output z
    def forward(self,x): # function to return ANN output
        return self.sigmoid(self.fc1(x)) # return output for input x
    
#load four data points for training
data = np.loadtxt('data.txt')
x = data[:,0] # Pull out x values for plotting later
y = data[:,1]
#### Convert numpy data into torch tensor format required for training
data = np.reshape(data,(1,4,2)) # reshape 4x2 mat of data into 1 batch
data = torch.tensor(data,dtype=torch.float) # convert numpy to torch tensor
labels = [0,0,1,1] # Ground truth labels ANN in to be trained with
labels = np.reshape(labels,(1,4,1)) # reshape array: 1 batch, 4 vals, 1 output
labels = torch.tensor(labels,dtype=torch.float) # dtype conversion
# Create data for entire x-y plane for evaluation
res = 50 # resolution of mesh grid
# Meshgrid of all x and y values in a  grid in the x-y plane
xGrid, yGrid = np.meshgrid(np.linspace(0,1,res),np.linspace(0,1,res))
# Take the two matrices in the mesh grid and place all x values in first column
# all y vals in second column
domainData = np.column_stack((np.reshape(xGrid,(res*res,)),np.reshape(yGrid,(res*res,))))
# Convert numpy data into torch data where the first dim has only 1 batch
domainData = torch.tensor(np.reshape(domainData,(1,res*res,2)),dtype = torch.float)
#### Declare and set up ANN
model = Net() # creates ANN object
criterion = torch.nn.MSELoss() # Set loss function
optimizer = torch.optim.SGD(model.parameters(),lr=7) # set learning rule
#### Loop to train ANN
for i in range(15):
    #### Standard code for iteration to updatate ANN parameters giving inputs 
    #### stored in vaiable data and the ground truth labels stored in variable 
    #### labels.If you use the same variable names in your code, then this 
    #### section will remain unchanged in yout soultion.
    optimizer.zero_grad()
    prediction = model(data)
    loss = criterion(prediction, labels)
    loss.backward()
    optimizer.step()
    #### End of standard code
    
    print(loss.item()) # disp error in training data
    #### Get ANN output for entire x,y domain and convert to numpy meshgrid
    output = model(domainData) # Predict output and convert to numpy meshgrid
    output = output.detach().numpy() # convert torch tensor to numpy array
    output = np.reshape(output,(res,res)) # reshape meshgrid
    print('o:  ',output)
    #### Plot ANN output and training data    
    fig = plt.figure(1)
    plt.clf()
    ax = fig.gca(projection='3d') # set axis to be 3d
    ax.plot_surface(xGrid,yGrid,output,cmap='cool') # Surface plot ANN output
    ax.scatter(x[0:2],y[0:2],[2,2],color='m') # training data for one class
    ax.scatter(x[2:4],y[2:4],[2,2],color='c') # training data for other class
    plt.xlabel('x')
    plt.ylabel('y')
    ax.view_init(90,-90) # rotate to look directly down suface
    plt.pause(0.001) # pause so figure updates
print(prediction) # print output, near zero for class 1, near unity for class 2


