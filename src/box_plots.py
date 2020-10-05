import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv('C:/Users/dylan/Work-Projects/ML74_/Ass2/iris.csv', index_col=0)
wine = pd.read_csv('C:/Users/dylan/Work-Projects/ML74_/Ass2/wine.csv', index_col=0)
digits = pd.read_csv('C:/Users/dylan/Work-Projects/ML74_/Ass2/digits.csv', index_col=0)
boston = pd.read_csv('C:/Users/dylan/Work-Projects/ML74_/Ass2/boston.csv', index_col=0)
diabetes = pd.read_csv('C:/Users/dylan/Work-Projects/ML74_/Ass2/diabetes.csv', index_col=0)
linnerud = pd.read_csv('C:/Users/dylan/Work-Projects/ML74_/Ass2/linnerud.csv', index_col=0)

list = [iris.iloc[0,:], iris.iloc[1,:], wine.iloc[0,:],wine.iloc[1,:], digits.iloc[0,:] , digits.iloc[1,:]]
plt.boxplot(list,  labels = ['iris SGD', 'iris Rprop', 'wine SGD', 'wine Rprop', 'digits SGD', 'digits Rprop'])
plt.title('box plot of accuracies for mutiple training runs')
plt.savefig('final_classes.png')
plt.show()

list = [boston.iloc[0,:], boston.iloc[1,:]]
list2 = [ diabetes.iloc[0,:],diabetes.iloc[1,:]]
list3 = [ linnerud.iloc[0,:] , linnerud.iloc[1,:]]
plt.boxplot(list, labels = ['boston SGD', 'boston Rprop'])
plt.title('box plot of room mean square error')
plt.savefig('final_reg1.png')
plt.show()
plt.boxplot(list2, labels =['diabetes SGD', 'diabetes Rprop'])
plt.title('box plot of room mean square error')
plt.savefig('final_reg2.png')
plt.show()
plt.boxplot(list3, labels = ['linnerud SGD', 'linnerud Rprop'])
plt.title('box plot of room mean square error')
plt.savefig('final_reg3.png')
plt.show()
