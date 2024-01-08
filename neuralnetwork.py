
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

#--------------------FUNCTIONS------------------------#

#ReLU activation function
def reLU(x):
    return np.maximum(0,x)

#ReLU derivative activation function
def reLU_der(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

#sigmoid activation function
def sigmoid(x):
    result = 1/(1+np.exp(-x))
    return result

#neural network functions
def forward(X):
    
    global h1,h2,output,z1,z2,z3
    
    #layer 1 
    z1 = np.dot(W1,X.T)
    h1 = reLU(z1)

    #layer 2
    h1 = h1.T
    new_col = np.ones(len(h1))
    h1 = np.insert(h1,0,new_col,axis = 1) 
    # print(h1.shape)
    z2 = np.dot(W2,h1.T)
    h2 = reLU(z2)
    
    #layer 3 (output)
    h2 = h2.T
    new_col = np.ones(len(h2))
    h2 = np.insert(h2,0,new_col,axis = 1) 
    z3 = np.dot(W3,h2.T)
    output = sigmoid(z3)
   
    return output

def compute_cost(output,y):
    N = len(y)
    
    cost_sum = np.sum(np.multiply(y,np.log(output)) + np.multiply(1-y, np.log(1-output))) #cross-entropy loss
    cost = -cost_sum/N
    
    return cost 

def backward(X,y,output):
    
    global W1,W2,W3

    #layer 3 (output)
    dJ_dz3 = -y + output 
    grad_w3 = np.dot(dJ_dz3,h2)
    
    W3_bar = np.delete(W3,0,1)
    grad_dz2 = np.multiply((reLU_der(z2)),(np.dot(W3_bar.T,dJ_dz3)))

    #layer 2
    grad_w2 = np.dot(grad_dz2,h1)
    
    W2_bar = np.delete(W2,0,1)
    grad_dz1 = np.multiply((reLU_der(z1)),(np.dot(W2_bar.T,grad_dz2)))
    
    #layer 1
    grad_w1 = np.dot(grad_dz1,X)
    
    #update parameters 
    W1 = W1 - alpha * grad_w1
    W2 = W2 - alpha * grad_w2
    W3 = W3 - alpha * grad_w3
    
    return W1, W2, W3

def predict(X,t):
    N = len(X)
   
    output = forward(X)
    output = output.T
    
    y = np.zeros(N)
    
    #the classifier that minimizes the misclassification rate
    for i in range(N):
        if(output[i]>=0.5):
            y[i]=1
    u = y - t
    err = np.count_nonzero(u)/N #misclassification rate
    #print("The misclassification rate is: " + str(err))      
    return err

#--------------------LOAD DATA------------------------# 

dataset = pd.read_csv('data_banknote_authentication.txt')
X = dataset.iloc[:, :-1].values
t = dataset.iloc[:, -1].values 

#Split data into training, validation and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size= 1/4, random_state = 5007)
X_train, X_valid, t_train, t_valid = train_test_split(X_train, t_train, test_size= 1/4, random_state = 5007) 

# print(X_train.shape)
# print(X_valid.shape)
# print(X_test.shape)

#Standardizing the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_valid[:, :] = sc.transform(X_valid[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

#finding size 
train_size = len(X_train)
valid_size = len(X_valid)
test_size = len(X_test)

#adding dummies to all sets
new_col = np.ones(train_size)
X_train = np.insert(X_train,0,new_col,axis = 1)

new_col = np.ones(valid_size)
X_valid = np.insert(X_valid,0,new_col,axis = 1)

new_col = np.ones(test_size)
X_test = np.insert(X_test,0,new_col,axis = 1)

#initialize parameters
epochs = 100
alpha = 0.001
inputSize = 4
n1 = 4
n2 = 4
outputSize = 1


#Neural Network Model
for D in range (2, inputSize+1): #iterating through D
    
    print("------------")
    print("D = " + str(D))
    print("------------")
    
    X1_train = X_train[:,:D+1]
    X1_valid = X_valid[:,:D+1]
    X1_test = X_test[:,:D+1]
    
    minValidError = []
    minW1 = []
    minW2 = []
    minW3 = []
    
    for i in range(2,n1+1): #iterating through n1           
        for j in range(2,n2+1): #iterating through n2
                    
            cost_training = []
            cost_valid = []
            training_misclass = []
            valid_misclass = []
            weights1 = []
            weights2 = []
            weights3 = []
            
            #initialize weights
            if D == 2: #to achieve smallest validation error
                W1 = np.full((i,D+1),0.3) #assign random weights 
                W2 = np.full((j,i+1),0.3) #assign random weights 
                W3 = np.full((outputSize,j+1),0.3) #assign random weights   
            
            else:
                W1 = np.full((i,D+1),0.7) #assign random weights 
                W2 = np.full((j,i+1),0.7) #assign random weights 
                W3 = np.full((outputSize,j+1),0.7) #assign random weights  
                
            for n in range(epochs):
                
                #shuffle data set
                X1_train, t_train = shuffle(X1_train, t_train, random_state = 5007)
                
                #forward pass on training set
                forward(X1_train)
                
                #calculate training cost
                cost1 = compute_cost(output,t_train)
                cost_training.append(cost1) 
                
                #backward pass on training set
                backward(X1_train,t_train,output)
                
                #calculate training misclassification error 
                err1 = predict(X1_train,t_train)
                training_misclass.append(err1)
                
                #forward pass on validation set using updated weights
                forward(X1_valid)
                
                #calculate validation misclassification cost
                cost2 = compute_cost(output,t_valid)
                cost_valid.append(cost2)
                
                #calculate validation error
                err2 = predict(X1_valid,t_valid)
                valid_misclass.append(err2)
                
                weights1.append(W1)
                weights2.append(W2)
                weights3.append(W3)
            
            minError = np.argmin(valid_misclass)
            print("For (" + str(i) + "," + str(j) + "), the smallest validation error is: " + str(min(valid_misclass)))
            minValidError.append(min(valid_misclass))
            minW1.append(weights1[minError])
            minW2.append(weights2[minError])
            minW3.append(weights3[minError])
            # print(len(minW1))
            # print(len(minW2))
            # print(len(minW3))
            
            if D == 2:
                plt.figure()
                plt.plot(cost_training, color = 'c', label = "Training Cross-Entropy Loss")
                plt.plot(cost_valid, color = 'b', label = "Validation Cross-Entropy Loss")
                plt.plot(training_misclass, color = 'r', label = "Training Misclassification Error")
                plt.plot(valid_misclass, color = 'g', label = "Validation Misclassification Error")
                plt.title('Learning Curve for D = 2, (' + str(i) + "," + str(j) + ')')
                plt.xlabel('Number of Epochs')
                plt.ylabel('Error')
                plt.ylim(0.1,1.1)
                plt.legend(prop={'size': 8})
                plt.show()
    
    print('')

    minTotalError = np.argmin(minValidError)

    W1 = minW1[minTotalError]
    W2 = minW2[minTotalError]
    W3 = minW3[minTotalError]

    n1_chosen = len(W1)
    n2_chosen = len(W2)
   
    print("The number of hidden units chosen is: (" + str(n1_chosen) + "," + str(n2_chosen) + ")")
    
    print('')
    
    print("The weights that achieve the smallest validation error is ")
    print(W1)
    print(W2)
    print(W3)
    
    print('')
    
    #calculate test misclassification error
    testError = predict(X1_test,t_test)
    print("The test error is: " + str(testError))

            