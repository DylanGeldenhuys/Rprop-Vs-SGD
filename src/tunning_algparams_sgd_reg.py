# Load datasets

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


def train_lr_SGD(criterion, train_loader,dataset_training, dataset_validation,lr,m,output = 1, epochs=100):
    lr_loss = {'training_loss': [], 'validation_loss': []} 
    for i in lr:
        torch.manual_seed(40)
        model = Net(dataset_training.X.shape[1], dataset_training.X.shape[1],output).double()
        optimizer = torch.optim.SGD(model.parameters(), lr = i, momentum= m)
        for epoch in range(epochs):
            for l, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                z = model(x).reshape((-1,))
                if output == 1:
                    z = model(x).reshape((-1,))
                else:
                    z = model(x)
  
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()

        if output == 1:
            lr_loss['training_loss'].append(criterion(model(dataset_training.X).reshape(-1,), dataset_training.y).data.item())
            lr_loss['validation_loss'].append(criterion(model(dataset_validation.X).reshape(-1,), dataset_validation.y).data.item())
        else:
            lr_loss['training_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
            lr_loss['validation_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
    return  lr_loss, lr


if __name__ == "__main__":

    # Regression
    diabetes_data = load_diabetes()
    boston_data = load_boston()
    linnerud_data = load_linnerud()
    data = diabetes_data

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30, random_state=40) 
    X_val, X_test, y_val, y_test =  train_test_split(X_test, y_test, test_size=0.4, random_state=40) 
    data_train = Data(X_train, y_train)
    data_val = Data(X_val, y_val)
    data_test = Data(X_test, y_test)

    criterion = nn.MSELoss()

    train_loader = DataLoader(dataset=data_train, batch_size=1)
    lr = np.arange(0.00001,0.007, 0.0005)
    print(lr)
    m = 0.1
    loss_lr,  Hparam_lr= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, 1,epochs=40 )
    m = 0.3
    loss_lr2,  Hparam_lr2= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, 1,epochs=40 )
    m = 0.5
    loss_lr3,  Hparam_lr3= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m,1, epochs=40 )
    m = 0.7
    loss_lr4, Hparam_lr4= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, 1, epochs=40 )
    m = 0.9
    loss_lr5,  Hparam_lr5= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m,1, epochs=40 )




    import matplotlib.pyplot as plt 

    plt.plot(Hparam_lr, loss_lr['training_loss'], label = 'train - momentum = 0.1')
    plt.plot(Hparam_lr, loss_lr['validation_loss'], label = 'val - momentum = 0.1')
    plt.plot(Hparam_lr2, loss_lr2['training_loss'], label = 'train - momentum = 0.3')
    plt.plot(Hparam_lr2, loss_lr2['validation_loss'], label = 'val - momentum = 0.3')
    plt.plot(Hparam_lr3, loss_lr3['training_loss'], label = 'train - momentum = 0.5')
    plt.plot(Hparam_lr3, loss_lr3['validation_loss'], label = 'val - momentum = 0.5')
    plt.plot(Hparam_lr4, loss_lr4['training_loss'], label = 'train - momentum = 0.7')
    plt.plot(Hparam_lr4, loss_lr4['validation_loss'], label = 'val - momentum = 0.7')
    plt.plot(Hparam_lr5, loss_lr5['training_loss'], label = 'train - momentum = 0.9')
    plt.plot(Hparam_lr5, loss_lr5['validation_loss'], label = 'val - momentum = 0.9')
    plt.legend(loc = 'upper left', prop={'size': 6})
    plt.title('Loss with varying learning rate and momentum on diabetes data')
    plt.savefig('C:/Users/dylan/Work-Projects/ML74_/Ass2/figures/hyperparam_tuning/SGD/mom_LR_diabetes.png')
    plt.show()
    
    plt.plot(Hparam_lr, loss_lr['training_loss'], label = 'training2')
    plt.plot(Hparam_lr, loss_lr['validation_loss'], label = 'validation2')
    plt.show()
    
