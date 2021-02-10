import numpy as np
import random

# load data set
data = np.loadtxt("hw1_train.dat.txt")

# set sign()
def sign(x):
    if x > 0:
        return 1
    else:
        return -1


result = []

def PLA():   #  PLA algorithm
    num = 11    # the vector length
    end = 0     # check whether finish or not
    count = 0   # record the number of updates
    i = 0       # point to the current data
    w = num*[0.0]   # weight vector 
    N = 100         # total data number
    x = num*[0.0]  # make x list initialize all 0.0
    
    while end < 5*N:
        i = random.randint(0, N-1)
        x[1:num] = np.multiply((num-1)*[4.0], data[i][0:num-1])                # replace vector x with data

        if sign(np.dot(w, x)) != data[i][-1]:       # find mistake
            y = num*[data[i][-1]]                     
            w += np.multiply(y, x)                  # update w to correct mistake
            end = 0
            count = count + 1
        else:
            end = end + 1

    return count

for j in range(0, 1000):
    result.append(PLA())

print(np.median(result))