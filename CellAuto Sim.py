import numpy as np
import math

"Initial parameters"
Iter1 = 100 #Number of runs
Size = 20 #Width of cell

def entropy(string):
        "Calculates the Shannon entropy of a string"
        # get probability of chars in string
        prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
        # calculate the entropy
        entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
        return entropy

def entropy_ideal(length):
        "Calculates the ideal Shannon entropy of a string with given length"
        prob = 1.0 / length
        return -1.0 * length * prob * math.log(prob) / math.log(2.0)

"Initial condition"
X = np.random.randint(2, size=Size) 
x = entropy(X.tolist())
print(X)
print(x)

"Rule for iteration"
R = ([1, 1, 1], [1, 1, 0],[1, 0, 1],[1, 0, 0],[0, 1, 1],[0, 1, 0],[0, 0, 1],[0, 0, 0])
R = np.array(R)
P = np.random.randint(2, size=8) #P[i] is the outcome for an R[i] neighborhood
P = np.array(P)

z = np.zeros((Iter1), int)
Z = np.concatenate([X, z])

"Iteration"
for i in range(0, Iter1):
    Y = np.zeros([1, (Size)], dtype=int)
    Y = np.array(Y).flatten()
    for j in xrange(len(X)): #Obtain neighborhood 
        if j == 0:
            num1 = X[(len(X)-1)]
        else:
            num1 = X[(j - 1)]
        num2 = X[j]
        if j == (len(X)-1):
            num3 = X[0]
        else:
            num3 = X[(j + 1)]
        num = np.array([num1,num2,num3])
        ans = (np.where((R == num).all(axis=1)))
        ans = np.array(ans)
        new = P[ans]
        new = new[0]
        nnum = np.array(new).flatten()
        Y[j] = Y[j] + nnum 
    X = Y #Next condition
    y = entropy(Y.tolist())
    print(Y)
    print(y)
    Z = np.concatenate([Z, Y])
    z = np.zeros((Size), int)
    Z = np.concatenate([Z, z])





