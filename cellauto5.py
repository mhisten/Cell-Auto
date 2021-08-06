import sys, os, codecs, csv, math
from math import log
from collections import Counter
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

path = '/Users/mhisten/Documents/python/cellauto/'
os.chdir(path)


def entropy(probability_list):
    """
    Calculates the entropy of a specified discrete probability distribution
    @input probability_list The discrete probability distribution
    """
    running_total = 0

    for item in probability_list:
        running_total += item * log(item, 2)

    if running_total != 0:
        running_total *= -1

    return running_total


def binary_entropy(p0, p1):
    """
    Calculates the binary entropy given two probabilities
    @input p0 Probability of first value
    @input p1 Probability of second value
    The two values must sum to 1.0
    """
    return entropy([p0, p1])


def matrix_entropy(matrix):
    """
    Calculates the "entropy" of a matrix by treating each element as
    independent and obtaining the histogram of element values
    @input matrix
    """
    counts = dict(Counter(matrix.flatten())).values()
    total_count = sum(counts)
    discrete_dist = [float(x) / total_count for x in counts]
    return entropy(discrete_dist)


def profile(matrices):
    """
    Calculates the "profile" (a list of the entropies) of a set of scaled
    filtered matrices as defined in the StackExchange answer
    @input matrices The set of scaled filtered matrices
    """
    return [matrix_entropy(scale) for scale in matrices]

def avg_components(component_matrix):
    running_total = 0
    num_components = 0

    for (row_num, col_num), value in np.ndenumerate(component_matrix):
        running_total += value
        num_components += 1

    output_value = running_total / num_components

    return output_value

def moving_window_filter(matrix, f, neighborhood_size):
    """
    Applies a filter function to a matrix using a neighborhood size
    @input matrix The matrix to apply the filter function to
    @input f The filter function, such as average, sum, etc.
    @input neighborhood_size The size of the neighborhood for the function
    application
    """
    matrix_height, matrix_width = matrix.shape

    output_matrix = np.zeros([matrix_height - neighborhood_size + 1,
                              matrix_width - neighborhood_size + 1])

    for (row_num, col_num), value in np.ndenumerate(matrix):
        # Check if it already arrived at the right-hand edge as defined by the
        # size of the neighborhood box
        if not ((row_num > (matrix_height - neighborhood_size) or
                col_num > (matrix_width - neighborhood_size))):
            # Obtain each pixel component of an (n x n) 2-dimensional matrix
            # around the input pixel, where n equals neighborhood_size
            component_matrix = np.zeros([neighborhood_size, neighborhood_size])

            for row_offset in range(0, neighborhood_size):
                for column_offset in range(0, neighborhood_size):
                    component_matrix[row_offset][column_offset] = \
                        matrix[row_num + row_offset][col_num + column_offset]

            # Apply the transformation function f to the set of component
            # values obtained from the given neighborhood
            output_matrix[row_num, col_num] = f(component_matrix)

    return output_matrix

def perms(n):
    if not n:
        return

    for i in range(2**n):
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s
        yield s

##Parameters

Iter1 = 40 #Symmetric matrix of cells and steps; must be greater than 32
n = 8 #number of bits
firstrow = np.random.randint(2, size=(Iter1)) #Initial condition
Matrix = perms(8) #Create all possible permutations

#Rule for iteration
R = ([1, 1, 1, 0, 0], [1, 1, 0, 0, 0],[1, 0, 1, 0, 0],[1, 0, 0, 0, 0],[0, 1, 1, 0, 0],[0, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 0, 0],[1, 1, 1, 1, 0], [1, 1, 0, 1, 0],[1, 0, 1, 1, 0],[1, 0, 0, 1, 0],[0, 1, 1, 1, 0],[0, 1, 0, 1, 0],[0, 0, 1, 1, 0],[0, 0, 0, 1, 0],
[1, 1, 1, 0, 1], [1, 1, 0, 0, 1],[1, 0, 1, 0, 1],[1, 0, 0, 0, 1],[0, 1, 1, 0, 1],[0, 1, 0, 0, 1],[0, 0, 1, 0, 1],[0, 0, 0, 0, 1],[1, 1, 1, 1, 1], [1, 1, 0, 1, 1],[1, 0, 1, 1, 1],[1, 0, 0, 1, 1],[0, 1, 1, 1, 1],[0, 1, 0, 1, 1],[0, 0, 1, 1, 1],[0, 0, 0, 1, 1])
R = np.array(R)

P=str('[1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 0]')
P=P.replace(' ','')
P=P.replace('[','')
P=P.replace(']','')

P = np.array(list(P), dtype=int)
#print((P))


Z2 = firstrow
Z3 = firstrow

for i in range(0,Iter1):
    z = np.array([])
    for j in range(0,(len(Z2))):
        if j == 0: #if first
            num1 = Z2[-2]
            num2 = Z2[-1]
            num3 = Z2[j]
            num4 = Z2[j+1]
            num5 = Z2[j+2]
        elif j == 1: #if second
            num1 = Z2[-1]
            num2 = Z2[j-1]
            num3 = Z2[j]
            num4 = Z2[j+1]
            num5 = Z2[j+2]
        elif j == (len(Z2)-1): #if last
            num1 = Z2[j-2]
            num2 = Z2[j-1]
            num3 = Z2[j]
            num4 = Z2[0]
            num5 = Z2[1]
        elif j == (len(Z2)-2): #if second last
            num1 = Z2[j-2]
            num2 = Z2[j-1]
            num3 = Z2[j]
            num4 = Z2[j+1]
            num5 = Z2[0]
        elif j < (len(Z2)-2):
            num1 = Z2[j-2]
            num2 = Z2[j-1]
            num3 = Z2[j]
            num4 = Z2[j+1]
            num5 = Z2[j+2]
        hood = np.array(([num1,num2,num3,num4,num5]), dtype=int)
        location = (np.where((R == hood).all(axis=1)))
        location = np.array(location)
        outcome = P[location]
        outcome = outcome[0]
        z = np.append(z, outcome)
    z = np.array(z, dtype=int) #next row
    Z2 = z 
    Z3 = np.vstack([Z3,z])


plt.imshow(Z3, cmap='tab20c',  interpolation='nearest')
#tab20c
#tab20
#copper
plt.axis('off')


plt.show()


