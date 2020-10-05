# Load datasets

from sklearn.datasets import load_diabetes, load_wine, load_digits
from sklearn.datasets import load_iris, load_boston
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
    return np.count_nonzero((yhat == data_set.y).numpy())/len(data_set)

def train_moving_avg(model, criterion, train_loader, val_loader, optimizer,dataset_training, dataset_validation, epochs=100):
    
    useful_stuff = {'training_loss': [], 'validation_loss': []} 
    epoch_loss = {'training_epoch_loss': [], 'validation_epoch_loss': []} 
    epoch_accuracy = {'training_epoch_accuracy': [], 'validation_epoch_accuracy': []} 
    epoch_loss = []
    for epoch in range(epochs):
        save_params = model.parameters()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
        Loss = criterion(model(dataset_training.X),dataset_training.y ).data.item()
        if epoch == 1:
            epoch_loss.append(Loss)
            epoch_loss.append(Loss)
            epoch_loss.append(Loss)
        epoch_loss.append(Loss)
        move_avg_val = np.mean(epoch_loss[-3:])
        std_val = np.std(epoch_loss)
        if np.mean(epoch_loss) > move_avg_val + std_val:
            model.parameters = save_params
            break
        #epoch_loss['training_epoch_loss'].append(accuracy(model,dataset_training))
        #epoch_loss['validation_epoch_loss'].append(accuracy(model,dataset_validation))
        print(criterion(model(dataset_training.X), dataset_training.y).data.item())
        epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
        epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
        epoch_accuracy['training_epoch_accuracy'].append(accuracy(model, dataset_training))
        epoch_accuracy['validation_epoch_accuracy'].append(accuracy(model, dataset_validation))
        


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


def train(model, criterion, train_loader, val_loader, optimizer,dataset_training, dataset_validation, epochs=100):
    
    useful_stuff = {'training_loss': [], 'validation_loss': []} 
    epoch_loss = {'training_epoch_loss': [], 'validation_epoch_loss': []} 
    epoch_accuracy = {'training_epoch_accuracy': [], 'validation_epoch_accuracy': []} 

    epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
    epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
    epoch_accuracy['training_epoch_accuracy'].append(accuracy(model, dataset_training))
    epoch_accuracy['validation_epoch_accuracy'].append(accuracy(model, dataset_validation))

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())

        epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
        epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
        epoch_accuracy['training_epoch_accuracy'].append(accuracy(model, dataset_training))
        epoch_accuracy['validation_epoch_accuracy'].append(accuracy(model, dataset_validation))
        correct = 0
    return  epoch_loss, epoch_accuracy

if __name__ == "__main__":
    # Classification
    iris_data = load_iris()
    wine_data = load_wine()
    digits_data = load_digits()
    data = digits_data # # NOTE : set data set to this variable

    lr = 0.08
    m = 0.5
    hl = [2,4,6, 8, 10, 11, 12, 13]
    # iter 1
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.30, random_state=40) 
    X_val, X_test, y_val, y_test =  train_test_split(X_test, y_test, test_size=0.4, random_state=40) 
    data_train = Data(X_train, y_train)
    data_val = Data(X_val, y_val)
    data_test = Data(X_test, y_test)

    final_acc = []
    for i in hl:
        torch.manual_seed(20)
        model = model = Net(data_train.X.shape[1], i,  len(np.unique(data_test.y))).double()
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(data_train, batch_size = len(data_train))
        optimizer = torch.optim.Rprop(model.parameters())#, lr , m)
        epoch_loss, epoch_acc = train(model, criterion, train_loader, train_loader, optimizer,data_train, data_val, epochs=60)
        plt.plot(epoch_acc['validation_epoch_accuracy'], label = '{} hidden units'.format(i))
        
        final_acc.append(epoch_acc['validation_epoch_accuracy'][-1])
        plt.legend()

    

    plt.title('Accuracy over epochs for varying hiddens units on digits data')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('C:/Users/dylan/Work-Projects/ML74_/Ass2/figures/hl_tuning/Rprop/acc_hl_digits.png')
    plt.show()
    plt.plot(hl, final_acc)
    plt.title('Validation accuracy for varying hidden units on digits data')
    plt.xlabel('hidden units')
    plt.ylabel('accuracy')
    plt.show()

    











