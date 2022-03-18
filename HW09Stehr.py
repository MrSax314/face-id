import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import cv2
import sympy
#1
x,x1,x2,x3,x4,y,y1,y2,y3,y4,u,v,b,E,ui,vi,bi,n = sympy.symbols('x x1 x2 x3 x4 y y1 y2 y3 y4 u v b E ui vi bi n')
sympy.init_printing(use_unicode=True)
z = (1+(sympy.exp(-(u*x+v*y+b))))**-1
E = ((0-z.subs({x:x1,y:y1}))**2)+((0-z.subs({x:x2,y:y2}))**2)+((1-z.subs({x:x3,y:y3}))**2)+((1-z.subs({x:x4,y:y4}))**2)

# Starting data and variable creation
data = np.loadtxt('data.txt') # load four data points for training
xdata = data[:,0] # Pull out x values for plotting later
ydata = data[:,1]
labels = [0,0,1,1] # Ground truth labels ANN in to be trained with
learning_rate = 7
n = learning_rate
uNext = np.random.random()#subs
vNext = np.random.random()
bNext = np.random.random()
zi = z.subs({u:uNext,v:vNext,b:bNext})
Ei = E.subs({u:uNext,v:vNext,b:bNext})
#print('E: ',E)
#print('Ei: ',E)
#ui,vi,bi,zi,n = sympy.symbols('ui vi bi zi n')
#### Update var functions ####
ui = u - (n * sympy.diff(E,u))
vi = v - (n * sympy.diff(E,v))
bi = b - (n * sympy.diff(E,b))
un = ui.subs({u:uNext,v:vNext,b:bNext})
vn = vi.subs({u:uNext,v:vNext,b:bNext})
bn = bi.subs({u:uNext,v:vNext,b:bNext})
#print('ui: ',ui,'\n')


'''
zout = zi.subs(x,1)
print(zout)
'''
#### Meshgrid creation ####
res = 50 # resolution of mesh grid
# Meshgrid of all x and y values in a  grid in the x-y plane
xGrid, yGrid = np.meshgrid(np.linspace(0,1,res),np.linspace(0,1,res))

#### Training ####

output = np.zeros((50,50))
for L in range(10): # 50
    uNow = uNext
    vNow = vNext
    bNow = bNext

    #### Plot ANN output and training data  ####  
    # fill output array (50x50) 
    for i in range(50):
        for j in range(50):
            output[i,j] = zi.subs({x:i,y:j})
            
    fig = plt.figure(1) # for graphing z evaluated on 2d plane
    plt.clf()
    ax = fig.gca(projection='3d') # set axis to be 3d
    ax.plot_surface(xGrid,yGrid,output,cmap='cool') # Surface plot ANN output
    ax.scatter(xdata[0:2],ydata[0:2],[2,2],color='m') # training data for one class
    ax.scatter(xdata[2:4],ydata[2:4],[2,2],color='c') # training data for other class
    plt.xlabel('x')
    plt.ylabel('y')
    ax.view_init(90,-90) # rotate to look directly down suface
    plt.pause(0.001) # pause so figure updates
    
    #### Update Values ####
    # check error and add updata u,v,b from z into zi again
    err = Ei.subs({x1:xdata[0],x2:xdata[1],x3:xdata[2],x4:xdata[3],y1:ydata[0],y2:ydata[1],y3:ydata[2],y4:ydata[3]})
    print('err: ',err)
    # update zi from z
    #print(uNow)
    uNext = ui.subs({u:uNow,v:vNow,b:bNow,x1:xdata[0],x2:xdata[1],x3:xdata[2],x4:xdata[3],y1:ydata[0],y2:ydata[1],y3:ydata[2],y4:ydata[3]})
    vNext = vi.subs({u:uNow,v:vNow,b:bNow,x1:xdata[0],x2:xdata[1],x3:xdata[2],x4:xdata[3],y1:ydata[0],y2:ydata[1],y3:ydata[2],y4:ydata[3]})
    bNext = bi.subs({u:uNow,v:vNow,b:bNow,x1:xdata[0],x2:xdata[1],x3:xdata[2],x4:xdata[3],y1:ydata[0],y2:ydata[1],y3:ydata[2],y4:ydata[3]})
    #print(uNext)
    '''
    uNext = uNext + uNow
    vNext = vNext + vNow
    bNext = bNext + bNow
    '''
    # Redefine eror functions based on updates equations
    # update equations
    zi = zi.subs({u:uNext,v:vNext,b:bNext})
    Ei = E.subs({u:uNext,v:vNext,b:bNext,x1:xdata[0],x2:xdata[1],x3:xdata[2],x4:xdata[3],y1:ydata[0],y2:ydata[1],y3:ydata[2],y4:ydata[3]})
    
    # handle u next and u now
'''
mpl_toolkits.mplot3d.axes3d.plot_surface
mpl_toolkits.mplot3d.axes3d.scatter
'''

#2
class Net(torch.nn.Module): # Cutsom ANN inherits from base class
    def __init__(self): # Default constuctor
        super(Net,self).__init__() # Run base class const
        self.hidden_size = 50
        self.fc1 = torch.nn.Linear(361,self.hidden_size) # Two in hidden out
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size,self.hidden_size) #hidden in one out
        self.sigmoid = torch.nn.Sigmoid() #Sigmoid for output z
        self.fc3 = torch.nn.Linear(self.hidden_size,1)
    def forward(self,x): # function to return ANN output
        #x = x.view(x.size(1), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) # return output for input x
        x = self.fc3(x)
        return x
    

#load face and nonFace img into data for training
pathFace = glob.glob("C:/Users/Avery Stehr/SpyderPythonScripts/Homework9/face/*")
pathNotFace = glob.glob("C:/Users/Avery Stehr/SpyderPythonScripts/Homework9/nonface/*")

#### data setup for training ####
im = cv2.imread(str(pathFace[0]),0) # read in grey scale jpgs
im = np.reshape(im,(1,361))
data = im # reshape to data list
for i in range(len(pathFace)-1):
    im = cv2.imread(str(pathFace[i+1]),0) # read in grey scale jpgs
    im = np.reshape(im,(1,361))
    data = np.vstack((data,im)) # reshape to data list
for i in range(len(pathNotFace)):
    im = cv2.imread(pathNotFace[i],0)
    im = np.reshape(im,(1,361)) # reshape 1x361 mat of data into 1 batch
    data = np.vstack((data,im)) # stack new im into data array
# Data formatting             
data_temp = data
data = np.reshape(data,(1,100,361)) # reshape 1 batch, 100 vals, 361 neurons
data = torch.tensor(data,dtype=torch.float) # convert numpy to torch tensor

#### Ground truths ####
labels = []
for i in range(100):
    if i < 50:
        labels.append(1) # Ground truth labels ANN in to be trained with
    else:
        labels.append(0)
# label formatting
labels = np.array(labels)
labels_temp = labels
labels = np.reshape(labels,(1,100,1)) # reshape array: 1 batch, 100 vals, 1 output
labels = torch.tensor(labels,dtype=torch.float) # dtype conversion

#### Declare and set up ANN ####
model = Net() # creates ANN object
criterion = torch.nn.MSELoss() # Set loss function
optimizer = torch.optim.SGD(model.parameters(),lr=0.005) # set learning rule .04

#### Loop to train ANN ####
runs = 2 # Number of runs though data 
(p,q) = np.shape(data_temp) # get number of jpg's in data to be checked
for i in range(p* runs):
    #### Standard code for iteration to updatate ANN parameters giving inputs 
    #### stored in vaiable data and the ground truth labels stored in variable 
    #### labels.If you use the same variable names in your code, then this 
    #### section will remain unchanged in yout soultion.
    #data = data.view(data.shape[1], -1)
    optimizer.zero_grad()
    prediction = model(data)
    loss = criterion(prediction, labels)
    loss.backward()
    optimizer.step()
    #### End of standard code
    
    #print(i, ")   Loss: ",loss.item()) # disp error in training data
    #print("prediction: ", prediction) # DEBUG
    #print("Loss: ",loss.item()) 
    #### Get ANN output for entire x,y domain and convert to numpy meshgrid
    

#print(prediction) # print output, near zero for class 1, near unity for class 2
pos = 0 # Starting position for jpg read in and test (pos = data[pos] for faces and data[pos+50] for non-faces)
bat_size = 4 # Size of batch for # of faces and not faces to be tested at 1 time
# pos 0 and batch 4 means re-test first 4 face jps and first 4 non face jpg w/ ANN
# pos shifts starting point of face and non-face selection
'''
for num in range(1):
    print('\n','faces in')
    # Test NN with face img
    for i in range(bat_size):
        testData = data_temp[i+pos,:] # Select any image from full 100x361 data set
        testData_temp = testData
        #cv2.imshow("test_jpg",np.reshape(testData_temp,(19,19)))
        #cv2.waitKey(3000) pause 3 sec
        #testData = np.column_stack((np.reshape(xGrid,(res*res,)),np.reshape(yGrid,(res*res,))))
        testData = torch.tensor(np.reshape(testData,(1,np.size(testData),1)),dtype = torch.float)
        output = model(testData.view(-1,19*19)) # Predict output and convert to numpy meshgrid
        output = output.detach().numpy() # convert torch tensor to numpy array
        print(output)
    print('Should be a 1^^^\n\nnot faces')
    # Test NN with non face img
    for i in range(bat_size):
        testData = data_temp[i+50+pos,:] # Select any image from full 100x361 data set
        testData_temp = testData
        #cv2.imshow("test_jpg",np.reshape(testData_temp,(19,19)))
        #cv2.waitKey(3000) pause 3 sec
        #testData = np.column_stack((np.reshape(xGrid,(res*res,)),np.reshape(yGrid,(res*res,))))
        testData = torch.tensor(np.reshape(testData,(1,np.size(testData),1)),dtype = torch.float)
        output = model(testData.view(-1,19*19)) # Predict output and convert to numpy meshgrid
        output = output.detach().numpy() # convert torch tensor to numpy array
        print(output)
    print("should be a 0^^^")
    pos += 4
    
'''