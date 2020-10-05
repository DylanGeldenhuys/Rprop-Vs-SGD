# Load datasets

from sklearn.datasets import load_iris, load_digits,load_wine
from sklearn.datasets import load_diabetes, load_boston
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Classification
diabetes_data = load_diabetes()
#breast_cancer_data = load_breast_cancer()
digits_data = load_digits()

# Regression
iris_data = load_iris()
boston_data = load_boston()
make_regression = np.load('C:/Users/dylan/Work-Projects/ML74_/Ass2/regression_data/make_regression.npy')

class Data():
    def __init__(self,X,y):
        self.X = torch.Tensor(X).double()
        self.y = torch.Tensor(y).long()

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

def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.X), 1)
    return (yhat == data_set.y).numpy().mean()


def train_lr_SGD(criterion, train_loader,dataset_training, dataset_validation,lr,m, epochs=100):
    loss_lr = {'training_loss': [], 'validation_loss': []} 
    accuracy_lr = {'training_acc': [], 'validation_acc': []} 
    losses = []
    for i in lr:
        torch.manual_seed(40)
        model = Net(dataset_training.X.shape[1], dataset_training.X.shape[1], len(np.unique(dataset_training.y))).double()

        optimizer = torch.optim.SGD(model.parameters(), lr = i, momentum= m)
  

        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
        loss_lr['training_loss'].append( criterion(model(dataset_training.X),dataset_training.y ).data.item() )
        loss_lr['validation_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
        accuracy_lr['training_acc'].append(accuracy(model, dataset_training))
        accuracy_lr['validation_acc'].append(accuracy(model, dataset_validation))
    return loss_lr, accuracy_lr, lr

if __name__ == "__main__":
    # Classification
    iris_data = load_iris()
    digits_data = load_digits()
    wine_data = load_wine()
    

    X_train, X_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.30, random_state=40) 

    X_val, X_test, y_val, y_test =  train_test_split(X_test, y_test, test_size=0.4, random_state=40) 
    data_train = Data(X_train, y_train)
    data_val = Data(X_val, y_val)
    data_test = Data(X_test, y_test)
    torch.manual_seed(60)
    criterion = nn.CrossEntropyLoss()
    print(data_train[1])
  
    
    train_loader = DataLoader(dataset=data_train, batch_size=1)

    lr = np.arange(0.001,0.005, 0.00002)
    print(lr)

    m = 0.1
    loss_lr, accuracy_lr, Hparam_lr= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, epochs=7 )
    print('done')
    m = 0.3
    loss_lr2, accuracy_lr2, Hparam_lr2= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, epochs=7 )
    print('done')
    m = 0.5
    loss_lr3, accuracy_lr3, Hparam_lr3= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, epochs=7 )
    print('done')
    m = 0.7
    loss_lr4, accuracy_lr4, Hparam_lr4= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, epochs=7 )
    print('done')
    m = 0.9
    loss_lr5, accuracy_lr5, Hparam_lr5= train_lr_SGD(criterion, train_loader,data_train,data_val,lr,m, epochs=7 )
    print('done')


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
    plt.title('Loss with varying learning rate and momentum on wine data')
    plt.savefig('C:/Users/dylan/Work-Projects/ML74_/Ass2/figures/hyperparam_tuning/SGD/mom_LR_wine.png')
    plt.show()
    
    plt.plot(Hparam_lr, loss_lr['training_loss'], label = 'training2')
    plt.plot(Hparam_lr, loss_lr['validation_loss'], label = 'validation2')
    plt.show()
    