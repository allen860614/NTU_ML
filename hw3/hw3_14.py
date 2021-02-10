import numpy as np
import math

# load data set
train_data = np.loadtxt("hw3_train.dat.txt")

X, Y = np.hsplit(train_data, [10])
X_0 = np.array(int(X.shape[0]) *[[1.0]])
X = np.hstack((X_0, X))

X = np.mat(X)
Y = np.mat(Y)
X_hat = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
W_LIN = np.dot(X_hat, Y)

Y_p = np.dot(X, W_LIN)

sum = 0
num = 0
for i in range(Y_p.shape[0]):
    sum += math.pow(Y_p[i]-Y[i], 2)
    num = num+1

print(sum/num)
