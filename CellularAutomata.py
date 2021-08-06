import numpy as np
from PIL import Image
from PIL import ImageOps
import math

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

Iter1 = 500 #Symmetric matrix of cells and steps
X = np.random.randint(2, size=(Iter1 + 1)) #Initial condition

'''
X = str('[0 0 0 0 0 1 0 0 0 0 0]')
X=X.replace(' ','')
X=X.replace('[','')
X=X.replace(']','')
X = np.array(list(X), dtype=int)
'''

#Rule for iteration
R = ([1, 1, 1], [1, 1, 0],[1, 0, 1],[1, 0, 0],[0, 1, 1],[0, 1, 0],[0, 0, 1],[0, 0, 0])
R = np.array(R)
P = np.random.randint(2, size=8) #P[i] is the outcome for an R[i] neighborhood

#P = (0,0,0,1,1,1,1,0) #Rule 30
#P = (0, 1, 1, 0, 1, 1, 1, 0)
#P = [1, 1, 1, 0, 0, 0, 0, 1]
#P = (1,0,0,1,0,1,0,1) 
#P = (1,0,0,1,0,1,1,1) 
#P = (0,1,1,0,1,0,1,0)
#P = (0, 1, 1, 0, 1, 1, 1, 0)



P = str('[1 0 1 0 1 0 0 1]')
P = str('[0 0 0 1 1 1 1 0]')
P = str('[1 0 0 1 0 1 1 1]')
P=P.replace(' ','')
P=P.replace('[','')
P=P.replace(']','')


P = np.array(list(P), dtype=int)
print((P))


z = np.zeros((Iter1), int)
Z1 = np.concatenate([X, z])
Z2 = X

for i in range(0, Iter1):
    Y = np.zeros([1, (Iter1 + 1)], dtype=int)
    Y = np.array(Y).flatten()
    for x in range(len(X)): #Obtain neighborhood 
        if x == 0:
            num1 = X[(len(X)-1)]
        else:
            num1 = X[(x - 1)]
        num2 = X[x]
        if x == (len(X)-1):
            num3 = X[0]
        else:
            num3 = X[(x + 1)]
        num = np.array([num1,num2,num3])
        ans = (np.where((R == num).all(axis=1)))
        ans = np.array(ans)
        new = P[ans]
        new = new[0]
        nnum = np.array(new).flatten()
        Y[x] = Y[x] + nnum 
    X = Y #Next condition
    y = entropy(Y.tolist())
    Z1 = np.concatenate([Z1, Y])
    z = np.zeros((Iter1), int)
    Z1 = np.concatenate([Z1, z])

indicator=0 #indicator function
Z2 = Z1
#print(Z2)
Z3 = []
while len(Z2)>0:
    if indicator==0:
        z3 = Z2[:(Iter1+1)]
        #print(type(z3))
        if len(Z3) == 0:
            Z3 = z3
        else:
            Z3 = np.concatenate((Z3, z3))
        Z2 = Z2[Iter1+1:]
        indicator = 1
    if indicator==1:
        Z2 = Z2[Iter1:]
        indicator = 0

Z3 = Z3.reshape(Iter1+1,Iter1+1)

Z = Z1.tolist()
dims = ((Iter1 * 2 + 1)),(Iter1 + 1)
i = Image.new("L", dims)
i.putdata(Z, 255)
i = i.crop(i.getbbox())
i = ImageOps.invert(i)
i.load()
i.show()
