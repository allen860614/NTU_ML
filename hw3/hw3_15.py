import numpy as np
import math
import random
from sklearn.metrics import mean_squared_error

# load data set
train_data = np.loadtxt("hw3_train.dat.txt")


# split
X, Y = np.hsplit(train_data, [10])
X_0 = np.array(int(X.shape[0]) *[[1.0]])
# add 0 column in X 
X = np.hstack((X_0, X))


X_hat = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
W_LIN = np.dot(X_hat, Y)

def squared_error(w, X, Y):
    sum = 0
    num = 0
    Y_p = np.dot(X, w)
    for i in range(Y_p.shape[0]):
        sum += math.pow(Y_p[i]-Y[i], 2)
        num = num+1
    return sum/num


def sgd(i, threshold, X, Y):
    random.seed(i)
    n = 0.001
    w = np.array(int(X.shape[1]) *[0.0])
    iter = 0
    while mean_squared_error(np.dot(X, w), Y)>threshold:   # use function in sklearn.metric to accelerate
        index = random.randint(0, X.shape[0]-1)
        w = w + n*2*(Y[index]-np.dot(w.T, X[index]))*X[index]
        iter = iter +1
    return iter


result = []
# calculate the threshold first to prevent the repeated computing.
threshold = 1.01*squared_error(W_LIN, X, Y)
for i in range(1000):
    result.append(sgd(i, threshold, X, Y))

print(np.mean(result))
