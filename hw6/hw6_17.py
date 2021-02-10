import numpy as np
import random

train_data = np.loadtxt("hw6_train.dat.txt")
test_data = np.loadtxt("hw6_test.dat.txt")
#print(train_data)

class Node:
    def __init__(self):
        self.i = None
        self.theta = None
        self.hyp = None
        self.left = None
        self.right = None


X, Y = np.hsplit(train_data, [-1])

pos_count = 0
neg_count = 0
for obj in Y:
    if obj[0]==1:
        pos_count+=1
    else:
        neg_count+=1


#print(X)
#print(sorted(feature))

li = []

def impurity(pos, neg):
    return (1-((pos/(pos+neg))*(pos/(pos+neg)) + (neg/(pos+neg))*(neg/(pos+neg))))

def same_point(a, b):
    for i in range(10):
        if a[i]!=b[i]:
            return False
        if i == 9 and a[-1]!=b[-1]:

            return True



class Tree:
    def train(self, train_data, root):
        #train_data = np.array(train_data)
        min_purity = 1000000000
        min_i = -1
        min_theta = -1
        pos_count = 0
        neg_count = 0
        
        for obj in train_data:
            if obj[-1]==1:
                pos_count+=1
            else:
                neg_count+=1
        #root = Node()

        if pos_count==0:
            root.hyp = -1
            #print(pos_count, neg_count)
            #li.append(root)
            return
        if neg_count==0:
            root.hyp = 1
            #li.append(root)
            return
        for i in range(len(train_data)-1):
            #k = same_point(train_data[i], train_data[i+1])
            if same_point(train_data[i], train_data[i+1]) == False:
                break
            if i == len(train_data)-1:
                if pos_count>neg_count:
                    root.hyp = 1
                else:
                    root.hyp = -1
                #.append(root)
                return
            
        for i in range(10):
            train_data = sorted(train_data, key = lambda s: s[i])
            X, Y = np.hsplit(np.array(train_data), [-1])
            left_pos = 0
            left_neg = 0
            
            for index in range(len(Y)-1):
                if Y[index][0] == 1:
                    left_pos+=1
                else:
                    left_neg+=1
                if X[index][i] == X[index+1][i]:
                    continue
                purity = (left_pos+left_neg)*impurity(left_pos, left_neg)+(pos_count-left_pos+neg_count-left_neg)*impurity(pos_count-left_pos, neg_count-left_neg)
                if purity<min_purity:
                    min_purity = purity
                    min_i = i
                    #print(X[index][i]+X[index+1])
                    min_theta = (X[index][i]+X[index+1][i])/2
        root.i = min_i
        root.theta = min_theta
        root.hyp = 0
        root.left = Node()
        root.right = Node()
        #li.append(root)
        left = []
        right = []
        #\print(min_theta)
        for obj in range(len(Y)):
            if X[obj][min_i] < min_theta:
                left.append(train_data[obj])
            else:
                right.append(train_data[obj])

        #print("min_i: {}, min_theta: {}".format(min_i, min_theta))
        self.train(np.array(left), root.left)

        self.train(np.array(right), root.right)
    
def test(data, root):
    #global li
    #i = 0
    while root!=None:
        if root.hyp==0:
            if data[root.i]<root.theta:
                root=root.left
            else:
                root=root.right
        else:
            return root.hyp



root = Node()

used = 1000*[False]
count = 0
Eout = 2000*[0]
#print(Ein)
X, Y = np.hsplit(test_data, [-1])
for i in range(2000):
    r = random.randint(0, len(train_data)-1)
    train_minus = []
    for j in range(int(len(train_data)/2)):
        r = random.randint(0, len(train_data)-1)
        train_minus.append(train_data[r])
        used[r] = True
    root = Node()

    Tree.train(Tree(), np.array(train_minus), root)

    for obj in range(len(X)):
        Eout[obj] += test(X[obj], root)
    #print(i, Ein[i])
        
for i in range(len(Y)):
    if Y[i]*Eout[i] <0 :
        count+=1
    #print(Y[i], Ein[i])

print(count/len(Y))