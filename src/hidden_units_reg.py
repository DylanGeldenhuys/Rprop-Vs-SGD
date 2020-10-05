
from sklearn.datasets import load_diabetes, load_wine, load_digits
from sklearn.datasets import load_iris, load_boston, load_linnerud
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
            loss_batch['training_batch_loss'].append(loss.data.item())
            loss_batch['validation_batch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())

        #epoch_loss['training_epoch_loss'].append(accuracy(model,dataset_training))
        #epoch_loss['validation_epoch_loss'].append(accuracy(model,dataset_validation))
        # NOTE: THIS IS OFR SINGLE OUTPUT
        #epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X).reshape(-1,), dataset_training.y).data.item())
        #epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X).reshape(-1,), dataset_validation.y).data.item())
        # NOTE: THIS IS FOR MULTIPLE OUTPUT
        epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
        epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
    return  epoch_loss, loss_batch




    return useful_stuff, epoch_loss, epoch_accuracy
def train_hl( criterion, dataset_training, dataset_validation, hl , lr, m ,optimizer = 'SGD', epochs=100):
    loss_hl = {'training_loss': [], 'validation_loss': []} 
    accuracy_hl = {'training_acc': [], 'validation_acc': []} 
    
    for  l in hl:
        torch.manual_seed(3)
        train_loader = DataLoader(dataset_training, batch_size = 1)
        model = Net(dataset_training.X.shape[1], l,  len(np.unique(dataset_training.y))).double()
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=m)
        if optimizer == 'Rprop':
            optimizer = torch.optim.Rprop(model.parameters())
        epoch_loss = []    
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
            epoch_loss.append(criterion(model(dataset_training.X),dataset_training.y ).data.item())
        plt.plot(epoch_loss)
        plt.show()
            

        loss_hl['training_loss'].append(criterion(model(dataset_training.X),dataset_training.y ).data.item())
        loss_hl['validation_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
        accuracy_hl['training_acc'].append(accuracy(model, dataset_training))
        accuracy_hl['validation_acc'].append(accuracy(model, dataset_validation))
        plt.plot(accuracy_hl['validation_acc'])
        plt.show()
    return loss_hl, accuracy_hl, hl




if __name__ == "__main__":
    # regression
    diabetes_data = load_diabetes()
    boston_data = load_boston()
    linnerud_data = load_linnerud()
    data = linnerud_data


    lr = 0.008
    m = 0.5
    hl = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # iter 1
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30, random_state=35) 
    X_val, X_test, y_val, y_test =  train_test_split(X_test, y_test, test_size=0.4, random_state=40) 
    data_train = Data(X_train, y_train)
    data_val = Data(X_val, y_val)
    data_test = Data(X_test, y_test)
#model, criterion, train_loader, optimizer,dataset_training, dataset_validation, epochs=100
    final_acc = []
    for i in hl:
        torch.manual_seed(20)
        model = model = Net(data_train.X.shape[1], i,  3).double()
        criterion = nn.MSELoss()
        train_loader = DataLoader(data_train, batch_size = len(data_train))
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=m )
        epoch_loss, epoch_acc = train(model, criterion, train_loader, optimizer,data_train, data_val, epochs=60)
        plt.plot( epoch_loss['validation_epoch_loss'], label = '{} hidden units'.format(i))
        #plt.plot(epoch_acc['training_epoch_accuracy'])
        final_acc.append(epoch_loss['validation_epoch_loss'][-1])
        plt.legend()

    

    plt.title('Accuracy over epochs for varying hiddens units on linnerud data')
    plt.xlabel('epochs')
    plt.ylabel('MSE loss')
    plt.savefig('C:/Users/dylan/Work-Projects/ML74_/Ass2/figures/hl_tuning/SGD/acc_hl_linnerud.png')
    plt.show()
    plt.plot(hl, final_acc)
    plt.title('Validation accuracy for varying hidden units on linnerud data')
    plt.xlabel('hidden units')
    plt.ylabel('MSE loss')
    plt.savefig('C:/Users/dylan/Work-Projects/ML74_/Ass2/figures/hl_tuning/SGD/acc_hl_linerrud2.png')
    plt.show()
