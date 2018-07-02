import numpy as np
import random
import matplotlib.pyplot as plt
from adaboost import Adaboost

def plot_results(X, Y, train_idx, test_idx, train_error, test_error):
    n = len(Y)
    plt.figure()
    nrows, ncols, index = 2,2,1
    for row in range(nrows):
        #Plot training and testing set
        idx = train_idx if row == 0 else test_idx
        plt.subplot(nrows,ncols,row*ncols+index)
        idx_label1 = [i for i in range(n) if i in idx and Y[i]==1]
        idx_label2 = [i for i in range(n) if i in idx and Y[i]!=1]
        plt.scatter(X[idx_label1,0], X[idx_label1,1], c='b')
        plt.scatter(X[idx_label2,0], X[idx_label2,1], c='r')
        plt.title('Training set') if row == 0 else plt.title('Testing set')
        plt.grid(True)
        
        #Plot training and testing error
        plt.subplot(nrows,ncols,row*ncols+index+1)
        plt.plot(train_error) if row == 0 else plt.plot(test_error)
        plt.axis([1, len(train_error), 0, 1])
        plt.title('Training error') if row == 0 else plt.title('Testing error')
        plt.xlabel('number of weak classifiers')
        plt.ylabel('classification error')
        plt.grid(True)
    plt.show()

def main():
    #Read in the dataset
    X = np.loadtxt('banknote_auth/data_banknote_auth.csv',delimiter=',')
    Y = np.loadtxt('banknote_auth/labels_banknote_auth.csv',dtype=str)
    
    #Map labels to {-1,1}
    labels = list(set(Y))
    Y = np.array([1 if y == labels[0] else -1 for y in Y])
    
    #Split the dataset
    training_ratio = 0.5 # Defines how much percentage of data to be used for training
    n = len(Y)
    perm = np.random.permutation(n)
    train_data_size = int(n * training_ratio)
    print(n)
    print(train_data_size)
    train_idx, test_idx = perm[:train_data_size], perm[train_data_size:]
    train_data = X[train_idx,:]
    train_label = Y[train_idx]
    test_data = X[test_idx,:]
    test_label = Y[test_idx]
    
    #Maximum number of weak learners to be used in Adaboost
    max_num_weak_learners = 69
    
    #Train and test error
    train_error = []
    test_error = []
    
    #Training Adaboost with weak learners
    model = Adaboost()
    for m in range(1,max_num_weak_learners+1):
        print("Training Adaboost with weak learners %d" % m)
        model.add_learner(train_data,train_label)
        train_error.append(model.prediction_error(train_data,train_label))
        test_error.append(model.prediction_error(test_data,test_label))
    
    print("Initial Training Error=%.4f  Testing Error= %.4f " % (train_error[0],test_error[0]))
    print("Final Training Error=%.4f  Testing Error= %.4f " % (train_error[-1],test_error[-1]))
    plot_results(X,Y,train_idx,test_idx,train_error,test_error)   

if __name__ == '__main__':
    main()