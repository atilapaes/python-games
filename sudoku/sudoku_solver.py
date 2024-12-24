#import pandas as pd

#read CSV file
#sample =pd.read_csv("sample1.csv")
import numpy as np
from numpy import genfromtxt
raw = genfromtxt('sample2.csv', delimiter=',',dtype=int)



def block_limits(block_number):
    """
    The 9x9 matrix is divided in 9 3x3 matrixes, that are named acording to the schema
    
    1,2,3
    4,5,6
    7,8,9
    
    
    Parameters
    ----------
    block_number : TYPE INT

    Returns
    -------
    Coordinates of the first (top-left) and last (bottom-right) elements 
    (from the 9x9 matrix)for each block
    
    """
    blocks=np.array([
    [[0,0],[2,2]],
    [[0,3],[2,5]],
    [[0,6],[2,8]],
    [[3,0],[5,2]],
    [[3,3],[5,5]],
    [[3,6],[5,8]],
    [[6,0],[8,2]],
    [[6,3],[8,5]],
    [[6,6],[8,8]]])
    
    return(blocks[block_number])


def create_workspace():
    """

    Returns
    -------
    Returns a 9x9 matrix where each element is a 1-d array with number form 0 to 9.
    This matrix represents all possible values for a value in the 9x9 matrix

    """
    workspace=np.array([
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]],
        [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
        
        ])
    
    return(workspace)
        

def remove_value_from_vec(vec,num):
    """
    Return a vector of same size, but the values num is changed by zero
    """
    vec[np.where(vec==num)[0][0]]=0
    return(vec)

def shrink_vec(vec):
    """
    Remove the zeros from a vector
    """
    return(np.trim_zeros(np.unique(vec)))


def num_possibilities(vec):
    """
    Return the number of possible values (not zero) we still have i a vector
    """
    return(shrink_vec(vec).size)


#=======    
workspace=create_workspace()
# for the array workspace: i is the row, j is the column, j is the vec element

# loop for the rows
for i in range(9):
    keys = shrink_vec(np.unique(raw[i,:])) # there are the values I need to remove from the first row
    for j in range(9):
        for key in keys:
            workspace[i,j]=remove_value_from_vec(workspace[i,j],key)
    




#view the file
#print(raw)

