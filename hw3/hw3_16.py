import numpy as np
import random

# load data set
train_data = np.loadtxt("hw3_train.dat.txt")

X, Y = np.hsplit(train_data, [10])
X_0 = np.array(int(X.shape[0]) *[[1.0]])
X = np.hstack((X_0, X))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sgd(i, X, Y):
    random.seed(i)
    n = 0.001
    w = np.array(int(X.shape[1]) *[0.0])
    iter = 0
    while iter<500:
        index = random.randint(0, X.shape[0]-1)
        w = w + n*sigmoid(-Y[index]*np.dot(w, X[index].T))*Y[index]*X[index]
        iter = iter +1

    sum = 0
    for i in range(X.shape[0]):
        sum += np.log(1+np.exp(-Y[i]*np.dot(w, X[i].T)))

    return sum/X.shape[0]



result = []
for i in range(1000):
    result.append(sgd(i, X, Y))

print(np.mean(result))



