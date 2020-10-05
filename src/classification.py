# Load datasets

from sklearn.datasets import load_diabetes, load_wine, load_digits
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

def train_(model, criterion, train_loader, val_loader, optimizer,dataset_training, dataset_validation, epochs=100):
    
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
        print(criterion(model(dataset_training.X), dataset_training.y).data.item())
        epoch_loss['training_epoch_loss'].append(criterion(model(dataset_training.X), dataset_training.y).data.item())
        epoch_loss['validation_epoch_loss'].append(criterion(model(dataset_validation.X), dataset_validation.y).data.item())
        epoch_accuracy['training_epoch_accuracy'].append(accuracy(model, dataset_training))
        epoch_accuracy['validation_epoch_accuracy'].append(accuracy(model, dataset_validation))
        


    return useful_stuff, epoch_loss, epoch_accuracy


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
    return useful_stuff, epoch_loss, epoch_accuracy

if __name__ == "__main__":
    # Classification
    iris_data = load_iris()
    wine_data = load_wine()
    digits_data = load_digits()

    
    
    X_train, X_test, y_train, y_test = train_test_split(digits_data.data, digits_data.target, test_size=0.30, random_state = 40) 
    X_val, X_test, y_val, y_test =  train_test_split(X_test, y_test, test_size=0.40, random_state = 40) 
    iris_data_train = Data(X_train, y_train)
    iris_data_val = Data(X_val, y_val)
    iris_data_test = Data(X_test, y_test)
    print(len(iris_data_train))
    print(len(iris_data_val))
  
    # define layers of NN
    input_size = iris_data_train.X.shape[1]
    hidden_layer_size = 7
    output_size = len(np.unique(iris_data_train.y))

    torch.manual_seed(11)
    model = Net(input_size, hidden_layer_size, output_size).double()
    
    optimizerSGD = torch.optim.SGD(model.parameters(), lr=0.007, momentum=0.5)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    
    train_loader = DataLoader(dataset=iris_data_train, batch_size=1)
    val_loader = DataLoader(dataset=iris_data_val, batch_size=1)
    
    useful_stuff_SGD, epoch_loss_SGD, epoch_acc_SGD = train(model,criterion, train_loader ,val_loader, optimizerSGD,iris_data_train,iris_data_test, epochs=50 )


    torch.manual_seed(11)
    input_size = iris_data_train.X.shape[1]
    hidden_layer_size = 12
    output_size = len(np.unique(iris_data_train.y))
    model = Net(input_size, hidden_layer_size, output_size).double()
    optimizerRprop = torch.optim.Rprop(model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset=iris_data_train, batch_size= len(iris_data_train))
   
    useful_stuff_Rprop, epoch_loss_Rprop, epoch_acc_Rprop = train(model,criterion, train_loader ,val_loader, optimizerRprop,iris_data_train,iris_data_test, epochs=50 )
    print('final', epoch_acc_SGD['training_epoch_accuracy'][-1])
    
    ########################################### iterations run ###################################
    
    ## 1st iteration ##
    lr = 0.005
    m = 0.7
    hidden_layer_size_sgd = 7
    hidden_layer_size_Rprop = 12
    torch.manual_seed(71)
    model = Net(input_size, hidden_layer_size_sgd, output_size).double()
    
    optimizerSGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=m)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    
    train_loader = DataLoader(dataset=iris_data_train, batch_size=1)
    val_loader = DataLoader(dataset=iris_data_val, batch_size=1)
    
    useful_stuff_SGD_1, epoch_loss_SGD_1, epoch_acc_SGD_1 = train(model,criterion, train_loader ,val_loader, optimizerSGD,iris_data_train,iris_data_test, epochs=60 )


    torch.manual_seed(71)
    input_size = iris_data_train.X.shape[1]
  
    output_size = len(np.unique(iris_data_train.y))
    model = Net(input_size, hidden_layer_size_Rprop, output_size).double()
    optimizerRprop = torch.optim.Rprop(model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset=iris_data_train, batch_size= len(iris_data_train))

    useful_stuff_Rprop_1, epoch_loss_Rprop_1, epoch_acc_Rprop_1 = train(model,criterion, train_loader ,val_loader, optimizerRprop,iris_data_train,iris_data_test, epochs=60 )

    ## 2nd iteration ##

    torch.manual_seed(99)
    model = Net(input_size, hidden_layer_size_sgd, output_size).double()
    
    optimizerSGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=m)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    
    train_loader = DataLoader(dataset=iris_data_train, batch_size=1)
    val_loader = DataLoader(dataset=iris_data_val, batch_size=1)
    
    useful_stuff_SGD_2, epoch_loss_SGD_2, epoch_acc_SGD_2 = train(model,criterion, train_loader ,val_loader, optimizerSGD,iris_data_train,iris_data_test, epochs=60 )


    torch.manual_seed(99)
    input_size = iris_data_train.X.shape[1]

    output_size = len(np.unique(iris_data_train.y))
    model = Net(input_size, hidden_layer_size_Rprop, output_size).double()
    optimizerRprop = torch.optim.Rprop(model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset=iris_data_train, batch_size= len(iris_data_train))

    useful_stuff_Rprop_2, epoch_loss_Rprop_2, epoch_acc_Rprop_2 = train(model,criterion, train_loader ,val_loader, optimizerRprop,iris_data_train,iris_data_test, epochs=60 )

    ## 3rd iteration ##

    torch.manual_seed(60)
    model = Net(input_size, hidden_layer_size_sgd, output_size).double()
    
    optimizerSGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=m)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    
    train_loader = DataLoader(dataset=iris_data_train, batch_size=1)
    val_loader = DataLoader(dataset=iris_data_val, batch_size=1)
    
    useful_stuff_SGD_3, epoch_loss_SGD_3, epoch_acc_SGD_3 = train(model,criterion, train_loader ,val_loader, optimizerSGD,iris_data_train,iris_data_test, epochs=60 )


    torch.manual_seed(60)
    input_size = iris_data_train.X.shape[1]
 
    output_size = len(np.unique(iris_data_train.y))
    model = Net(input_size, hidden_layer_size_Rprop, output_size).double()
    optimizerRprop = torch.optim.Rprop(model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset=iris_data_train, batch_size= len(iris_data_train))

    useful_stuff_Rprop_3, epoch_loss_Rprop_3, epoch_acc_Rprop_3 = train(model,criterion, train_loader ,val_loader, optimizerRprop,iris_data_train,iris_data_test, epochs=60 )

    ## 4th iteration ##

    torch.manual_seed(30)
    model = Net(input_size, hidden_layer_size_sgd, output_size).double()
    
    optimizerSGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=m)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    
    train_loader = DataLoader(dataset=iris_data_train, batch_size=1)
    val_loader = DataLoader(dataset=iris_data_val, batch_size=1)
    
    useful_stuff_SGD_4, epoch_loss_SGD_4, epoch_acc_SGD_4 = train(model,criterion, train_loader ,val_loader, optimizerSGD,iris_data_train,iris_data_test, epochs=60 )


    torch.manual_seed(30)
    input_size = iris_data_train.X.shape[1]

    output_size = len(np.unique(iris_data_train.y))
    model = Net(input_size, hidden_layer_size_Rprop, output_size).double()
    optimizerRprop = torch.optim.Rprop(model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset=iris_data_train, batch_size= len(iris_data_train))

    useful_stuff_Rprop_4, epoch_loss_Rprop_4, epoch_acc_Rprop_4 = train(model,criterion, train_loader ,val_loader, optimizerRprop,iris_data_train,iris_data_test, epochs=60 )

    ## 5th iteration ##

    
    torch.manual_seed(20)
    model = Net(input_size, hidden_layer_size_sgd, output_size).double()
    
    optimizerSGD = torch.optim.SGD(model.parameters(), lr=lr, momentum=m)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    
    train_loader = DataLoader(dataset=iris_data_train, batch_size=1)
    val_loader = DataLoader(dataset=iris_data_val, batch_size=1)
    
    useful_stuff_SGD_5, epoch_loss_SGD_5, epoch_acc_SGD_5 = train(model,criterion, train_loader ,val_loader, optimizerSGD,iris_data_train,iris_data_test, epochs=60 )


    torch.manual_seed(20)
    input_size = iris_data_train.X.shape[1]

    output_size = len(np.unique(iris_data_train.y))
    model = Net(input_size, hidden_layer_size_Rprop, output_size).double()
    optimizerRprop = torch.optim.Rprop(model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset=iris_data_train, batch_size= len(iris_data_train))

    useful_stuff_Rprop_5, epoch_loss_Rprop_5, epoch_acc_Rprop_5 = train(model,criterion, train_loader ,val_loader, optimizerRprop,iris_data_train,iris_data_test, epochs=60 )

    import matplotlib.pyplot as plt 
    # come back to this
    collect_SGD = [epoch_acc_SGD_1['validation_epoch_accuracy'][-1],epoch_acc_SGD_2['validation_epoch_accuracy'][-1], epoch_acc_SGD_3['validation_epoch_accuracy'][-1], epoch_acc_SGD_4['validation_epoch_accuracy'][-1], epoch_acc_SGD_5['validation_epoch_accuracy'][-1]]
    collect_Rprop = [epoch_acc_Rprop_1['validation_epoch_accuracy'][-1],epoch_acc_Rprop_2['validation_epoch_accuracy'][-1], epoch_acc_Rprop_3['validation_epoch_accuracy'][-1], epoch_acc_Rprop_4['validation_epoch_accuracy'][-1], epoch_acc_Rprop_5['validation_epoch_accuracy'][-1]]
    
    collect_iris = pd.DataFrame([collect_SGD, collect_Rprop])
    collect_iris.to_csv('digits.csv')
 
    plt.figure
    plt.plot(epoch_loss_Rprop['training_epoch_loss'], label = 'training')
    plt.plot(epoch_loss_Rprop['validation_epoch_loss'], label = 'validation')
    plt.title('Rprop loss per epoch')
    plt.show()


    plt.figure()
    plt.plot(epoch_loss_Rprop['training_epoch_loss'], label = 'train - Rprop',color = 'b', linewidth=0.5)
    plt.plot(epoch_loss_Rprop['validation_epoch_loss'], label = 'val - Rprop', color = 'b')
    plt.plot(epoch_loss_SGD['training_epoch_loss'], label = 'train - SGD', color = 'y',linewidth=0.5)
    plt.plot(epoch_loss_SGD['validation_epoch_loss'], label = 'val - SGD', color = 'y')
    plt.legend()
    plt.title('SGD vs Rprop loss on digits data')
    plt.xlabel('epoch')
    plt.ylabel('Cross entropy loss')
    plt.show()

    plt.figure()
    plt.plot(epoch_acc_Rprop['validation_epoch_accuracy'], label = 'val - Rprop', color = 'b')
    plt.plot(epoch_acc_Rprop['training_epoch_accuracy'], label = 'train - Rprop',color = 'b', linewidth=0.5)
    plt.plot(epoch_acc_SGD['validation_epoch_accuracy'], label = 'val - SGD', color = 'y')
    plt.plot(epoch_acc_SGD['training_epoch_accuracy'], label = 'train - SGD',color = 'y',linewidth=0.5)
    plt.legend()
    plt.title('SGD vs Rprop loss on digits data')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.show()
    print(max(epoch_acc_SGD['validation_epoch_accuracy']))
    print(max(epoch_acc_Rprop['validation_epoch_accuracy']))

    











