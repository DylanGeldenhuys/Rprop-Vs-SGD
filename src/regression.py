from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits
from sklearn.datasets import load_iris, load_boston, load_linnerud
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Data():
    def __init__(self,X,y):
        self.X = torch.Tensor(X).double()
        self.y = torch.Tensor(y).double()
    def __getitem__(self, index):          
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)

class Net(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.linear2=nn.Linear(H,D_out)

        
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))  
        x=self.linear2(x)
        return x
class linear_regression(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat

def train_model_BGD(iter):
    for epoch in range(iter):
        for x,y in trainloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(model, loss.tolist())          
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        get_surface.plot_ps()

def train(model, criterion, train_loader, optimizer,dataset_training, dataset_validation, epochs=100):
    

    epoch_loss = {'training_epoch_loss': [], 'validation_epoch_loss': []} 
    epoch_accuracy = {'training_epoch_accuracy': [], 'validation_epoch_accuracy': []} 
    loss_batch = {'training_batch_loss': [], 'validation_batch_loss': []} 

    epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
    epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            # NOTE: THIS IS FOR SINGLE OUPUT
            #z = model(x).reshape((-1,))
            
            # NOTE: THIS IS FOR MULTI OUPUT
            z = model(x)
            #print(z.size())
            #print(y.size())
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            # NOTE: THIS IS FOR SINGLE OUTPUT
            #loss_batch['training_batch_loss'].append(loss.data.item())
            #loss_batch['validation_batch_loss'].append(criterion(model(dataset_validation.X).reshape(-1,), dataset_validation.y).data.item())
            # NOTE: THIS IS FOR MULTI OUTPUT
            #loss_batch['training_batch_loss'].append(loss.data.item())
            #loss_batch['validation_batch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())

        #epoch_loss['training_epoch_loss'].append(accuracy(model,dataset_training))
        #epoch_loss['validation_epoch_loss'].append(accuracy(model,dataset_validation))
        # NOTE: THIS IS OFR SINGLE OUTPUT
        #epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X).reshape(-1,), dataset_training.y).data.item())
        #epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X).reshape(-1,), dataset_validation.y).data.item())
        # NOTE: THIS IS FOR MULTIPLE OUTPUT
        epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
        epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
    return  epoch_loss, loss_batch


if __name__ == "__main__":
    # Regression
    diabetes_data = load_diabetes()
    boston_data = load_boston()
    #make_regression = np.load('C:/Users/dylan/Work-Projects/ML74_/Ass2/regression_data/make_regression.npy')
    linnerud_data = load_linnerud()
    data = diabetes_data

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30, random_state=40) 
    X_val, X_test, y_val, y_test =  train_test_split(X_test, y_test, test_size=0.4, random_state=40) 
    data_train = Data(X_train, y_train)
    data_val = Data(X_val, y_val)
    data_test = Data(X_test, y_test)
    criterion = nn.MSELoss()

    input_size = data_train.X.shape[1]
    hidden_layer_size = 7#input_size 
    output_size = 1
    torch.manual_seed(11)
    model2 = Net(input_size, hidden_layer_size, output_size).double()
    optimizer = torch.optim.SGD(model2.parameters(), lr = 0.005, momentum=0.3)
    trainloader = DataLoader(dataset = data_train, batch_size = 1)
    epoch_loss_SGD, batch_loss_SGD = train(model2, criterion, trainloader,optimizer, data_train, data_val, epochs = 60)

    input_size = data_train.X.shape[1]
    hidden_layer_size = 7#input_size 
   
    torch.manual_seed(11)
    model = Net(input_size, hidden_layer_size, output_size).double()
    optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)
    trainloader = DataLoader(dataset = data_train, batch_size = len(data_train))
    epoch_loss_Rprop, batch_loss_Rprop = train(model, criterion, trainloader,optimizer, data_train, data_val, epochs = 60)

    
    ############################ iter 1 ############################

    input_size = data_train.X.shape[1]
    hidden_layer_size_sgd = 7#input_size 
    hidden_layer_size_rprop = 7
    lr = 0.001
    m = 0.3
    output_size = 1

    torch.manual_seed(11)
    model2 = Net(input_size, hidden_layer_size_sgd, output_size).double()
    optimizer = torch.optim.SGD(model2.parameters(), lr = lr, momentum=m)
    trainloader = DataLoader(dataset = data_train, batch_size = 1)
    epoch_loss_SGD_1, batch_loss_SGD_1 = train(model2, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    input_size = data_train.X.shape[1]
    hidden_layer_size = 1#input_size 
    torch.manual_seed(11)
    model = Net(input_size, hidden_layer_size_rprop, output_size).double()
    optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)
    trainloader = DataLoader(dataset = data_train, batch_size = len(data_train))
    epoch_loss_Rprop_1, batch_loss_Rprop_1 = train(model, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    ########################### iter 2
    torch.manual_seed(20)
    model2 = Net(input_size, hidden_layer_size_sgd, output_size).double()
    optimizer = torch.optim.SGD(model2.parameters(), lr = lr, momentum=m)
    trainloader = DataLoader(dataset = data_train, batch_size = 1)
    epoch_loss_SGD_2, batch_loss_SGD_2 = train(model2, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    input_size = data_train.X.shape[1]
    hidden_layer_size = 1#input_size 
    torch.manual_seed(20)
    model = Net(input_size, hidden_layer_size_rprop, output_size).double()
    optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)
    trainloader = DataLoader(dataset = data_train, batch_size = len(data_train))
    epoch_loss_Rprop_2, batch_loss_Rprop_2 = train(model, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    ##################### iter 3

    torch.manual_seed(30)
    model2 = Net(input_size, hidden_layer_size_sgd, output_size).double()
    optimizer = torch.optim.SGD(model2.parameters(), lr = lr, momentum=m)
    trainloader = DataLoader(dataset = data_train, batch_size = 1)
    epoch_loss_SGD_3, batch_loss_SGD_3 = train(model2, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    input_size = data_train.X.shape[1]
    hidden_layer_size = 1#input_size 

    torch.manual_seed(30)
    model = Net(input_size, hidden_layer_size_rprop, output_size).double()
    optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)
    trainloader = DataLoader(dataset = data_train, batch_size = len(data_train))
    epoch_loss_Rprop_3, batch_loss_Rprop_3 = train(model, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    ###################### iter 4

    torch.manual_seed(40)
    model2 = Net(input_size, hidden_layer_size_sgd, output_size).double()
    optimizer = torch.optim.SGD(model2.parameters(), lr = lr, momentum=m)
    trainloader = DataLoader(dataset = data_train, batch_size = 1)
    epoch_loss_SGD_4, batch_loss_SGD_4 = train(model2, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    input_size = data_train.X.shape[1]
    hidden_layer_size = 1#input_size 

    torch.manual_seed(40)
    model = Net(input_size, hidden_layer_size_rprop, output_size).double()
    optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)
    trainloader = DataLoader(dataset = data_train, batch_size = len(data_train))
    epoch_loss_Rprop_4, batch_loss_Rprop_4 = train(model, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)


    ########################### iter 5

    torch.manual_seed(50)
    model2 = Net(input_size, hidden_layer_size_sgd, output_size).double()
    optimizer = torch.optim.SGD(model2.parameters(), lr = lr, momentum=m)
    trainloader = DataLoader(dataset = data_train, batch_size = 1)
    epoch_loss_SGD_5, batch_loss_SGD_5 = train(model2, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    input_size = data_train.X.shape[1]
    hidden_layer_size = 1#input_size 

    torch.manual_seed(50)
    model = Net(input_size, hidden_layer_size_rprop, output_size).double()
    optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)
    trainloader = DataLoader(dataset = data_train, batch_size = len(data_train))
    epoch_loss_Rprop_5, batch_loss_Rprop_5 = train(model, criterion, trainloader,optimizer, data_train, data_test, epochs = 40)

    collect_rprop = [ epoch_loss_Rprop_1['validation_epoch_loss'][-1], epoch_loss_Rprop_2['validation_epoch_loss'][-1], epoch_loss_Rprop_3['validation_epoch_loss'][-1], epoch_loss_Rprop_4['validation_epoch_loss'][-1], epoch_loss_Rprop_5['validation_epoch_loss'][-1]]
    collect_sgd = [epoch_loss_SGD_1['validation_epoch_loss'][-1], epoch_loss_SGD_2['validation_epoch_loss'][-1], epoch_loss_SGD_3['validation_epoch_loss'][-1], epoch_loss_SGD_4['validation_epoch_loss'][-1] , epoch_loss_SGD_5['validation_epoch_loss'][-1]]
    #print(epoch_loss_Rprop_1['validation_epoch_loss'][-1])
    #collect_rprop = [ epoch_loss_Rprop_1['validation_epoch_loss'][-1], epoch_loss_Rprop_2['validation_epoch_loss'][-1] ]#, epoch_loss_Rprop_3['validation_epoch_loss'][-1], epoch_loss_Rprop_4['validation_epoch_loss'][-1], epoch_loss_Rprop_5['validation_epoch_loss'][-1]]
    #$print(epoch_loss_SGD_1['validation_epoch_loss'][-1])
    #, epoch_loss_SGD_2['validation_epoch_loss'][-1] ]#, epoch_loss_SGD_3['validation_epoch_loss'][-1], epoch_loss_SGD_4['validation_epoch_loss'][-1], epoch_loss_SGD_5['validation_epoch_loss'][-1]]


    Collect = pd.DataFrame([collect_sgd, collect_rprop])
    Collect.to_csv('diabetes.csv')
    
    import matplotlib.pyplot as plt
    #plt.plot( X.numpy().T[0], Y.numpy().T[0])
    #plt.plot(X.numpy().T[0], model(X.double()).detach().numpy())
    #plt.show()
    plt.figure()
    plt.plot(np.sqrt(epoch_loss_Rprop['training_epoch_loss']), label = 'train - Rprop',color = 'b', linewidth=0.5)
    plt.plot(np.sqrt(epoch_loss_Rprop['validation_epoch_loss']), label = 'val - Rprop', color = 'b')
    plt.plot(np.sqrt(epoch_loss_SGD['training_epoch_loss']), label = 'train - SGD', color = 'y',linewidth=0.5)
    plt.plot(np.sqrt(epoch_loss_SGD['validation_epoch_loss']), label = 'val - SGD', color = 'y')
    plt.legend()
    plt.title('SGD vs Rprop loss on diabetes data')
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    #plt.savefig('C:/Users/dylan/Work-Projects/ML74_/Ass2/figures/sgd_rprop_loss_diabetes.png')
    print(min(np.sqrt(epoch_loss_Rprop['validation_epoch_loss'])))
    print(min(np.sqrt(epoch_loss_SGD['validation_epoch_loss'])))
    
    
    plt.show()


    