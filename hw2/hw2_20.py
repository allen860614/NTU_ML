import numpy as np
import random

def sign(x):
    if x<=0:
        return -1
    else:
        return 1


data_size = 200
tou = 0.1
T = 10000
result = []

# generate input data
def gen_input():
    x = []
    y = []
    for i in range(data_size):
        x.append(random.uniform(-1, 1))
    x.sort()
    for i in range(data_size):
        if random.random()>=tou:
            y.append(sign(x[i]))
        else:
            y.append(-sign(x[i]))
    return x, y


def calc_Ein(x, y):
    thetas = [float(-1)] + [(x[i]+x[i + 1])/2 for i in range(data_size-1)]
    Ein = len(x)
    target_theta = 0.0
    x = np.array(x)
    y = np.array(y)
    for theta in thetas:
        y_pos = np.where(x > theta, 1, -1)
        y_neg = np.where(x < theta, 1, -1)
        error_pos = sum(y_pos != y)
        error_neg = sum(y_neg != y)
        if error_pos > error_neg:
            if Ein > error_neg:
                Ein = error_neg
                target_theta = theta
        else:
            if Ein > error_pos:
                Ein = error_pos
                target_theta = theta
    return Ein, target_theta


for i in range(T):
    x, y = gen_input()
    curr_Ein, theta = calc_Ein(x, y)
    Ein = curr_Ein/data_size
    Eout = 1/2 * (abs(theta))*(1-2*tou)+tou
    result.append(Eout-Ein)
print(np.mean(result))
