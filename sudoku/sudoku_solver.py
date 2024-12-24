#import pandas as pd

#read CSV file
#sample =pd.read_csv("sample1.csv")
import numpy as np
from numpy import genfromtxt
raw = genfromtxt('sample2.csv', delimiter=',',dtype=int)
solution=raw


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
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]]
        
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
    

def remove_others_than(vec,num):
    # change all values in a vector to zero except the value num
    for i in range(9):
        if i != num:
            vec[i]=0
    return(vec)    


def initial_conditions(ws):
    # Step 1: write the initial conditions from raw to workspace
    for i in range(9):
        for j in range(9):
            if raw[i,j] !=0:
                ws[i,j]= remove_others_than(ws[i,j],raw[i,j])
    return(ws)

def loop_rows(ws,raw):
    # loop for the rows
    for i in range(9):
        keys = shrink_vec(np.unique(raw[i,:])) # there are the values I need to remove from the first row
        for j in range(9):
            for key in keys:
                if num_possibilities(ws[i,j]) >1 and key in ws[i,j]: #otherwise, it will remove the key from its native place
                    ws[i,j]=remove_value_from_vec(ws[i,j],key)
    return(ws)


def loop_columns(ws,raw):
    #loop for the columns
    for j in range(9):
        keys = shrink_vec(np.unique(raw[:,j])) # there are the values I need to remove from the first row
        for i in range(9):
            for key in keys:
                if num_possibilities(ws[i,j]) >1 and key in ws[i,j]: #otherwise, it will remove the key from its native place
                    ws[i,j]=remove_value_from_vec(ws[i,j],key)    
    return(ws)

#=======    
# Aplly firt checking, with all the rows ans columns

#step zero: create the matrix of possible keys
workspace=create_workspace()
print('step 0 done. N zeros:',np.count_nonzero(workspace==0))

#step 1: apply the initial condition of the sudoku
workspace=initial_conditions(workspace)
print('step 1 done. N zeros:',np.count_nonzero(workspace==0))

#step 2: loop over rows, and remove possible values for the keys present in the row
workspace=loop_rows(workspace,raw)
print('step 2 done. N zeros:',np.count_nonzero(workspace==0))

#step 3: loop over columns, and remove possible values for the keys present in the columns
workspace=loop_columns(workspace,raw)
print('step 3 done. N zeros:',np.count_nonzero(workspace==0))



for i in range(9):
    for j in range(9):
        if num_possibilities(workspace[i,j])==1 and raw[i,j] ==0:
            print(i,j)
# any output here would bring a new key value
