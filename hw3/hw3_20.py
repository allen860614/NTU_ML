import numpy as np
import math

# load data set
train_data = np.loadtxt("hw3_train.dat.txt")
test_data = np.loadtxt("hw3_test.dat.txt")

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def transform(x, q):
    t = np.multiply(x, x)
    r = x
    for i in range(q-1):
        r = np.hstack((r, t))
        t = np.multiply(t, x)
    return r


X, Y = np.hsplit(train_data, [10])
X_test, Y_test = np.hsplit(test_data, [10])
X_0 = np.array(int(X.shape[0]) *[[1.0]])
X = transform(X, 10)
X_test = transform(X_test, 10)
X = np.hstack((X_0, X))
X_0 = np.array(int(X_test.shape[0])*[[1.0]])
X_test = np.hstack((X_0, X_test))

X = np.mat(X)
Y = np.mat(Y)
X_hat = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
W_LIN = np.dot(X_hat, Y)

Y_p = np.dot(X, W_LIN)

Ein = 0
num = 0

for i in range(Y_p.shape[0]):
    if sign(Y_p[i])!= Y[i]:
        Ein += 1
    num = num+1
Ein = Ein/num
Eout = 0
num = 0
Y_p = np.dot(X_test, W_LIN)
for i in range(Y_p.shape[0]):
    if sign(Y_p[i])!= Y_test[i]:
        Eout += 1
    num = num+1
Eout = Eout/num
print(abs(Ein-Eout))


