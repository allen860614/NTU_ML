import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from liblinearutil import *

train_data = np.loadtxt("hw4_train.dat.txt")
test_data = np.loadtxt("hw4_test.dat.txt")

print(train_data.shape)
X, Y = np.hsplit(train_data, [6])
print(X.shape)
poly = PolynomialFeatures()
X = poly.fit_transform(X)
print(X.shape)

y, x = svm_read_problem('../heart_scale')
m = train(y[:200], x[:200], '-c 4')
p_label, p_acc, p_val = predict(y[200:], x[200:], m)