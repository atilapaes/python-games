#import pandas as pd

#read CSV file
#sample =pd.read_csv("sample1.csv")
import numpy as np
from numpy import genfromtxt
raw = genfromtxt('sample2.csv', delimiter=',',dtype=int)
solution=raw


def block_limits(block):
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
    
    return(blocks[block])


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


def num_options(vec):
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

def update_sol(ws,sol):
    for i in range(9):
        for j in range(9):
            if sol[i,j] == 0 and num_options(ws[i,j])==1:
                sol[i,j]=shrink_vec(ws[i,j])[0]
                print('new key found in',i,j,',as',shrink_vec(ws[i,j])[0])
    return(sol)

def initial_conditions(ws,sol):
    # Step 1: write the initial conditions from raw to workspace
    for i in range(9):
        for j in range(9):
            if sol[i,j] !=0:
                ws[i,j]= remove_others_than(ws[i,j],sol[i,j])
            
    return(ws)


def loop_rows(ws,sol):
    # loop for the rows
    for i in range(9):
        keys = shrink_vec(np.unique(sol[i,:])) # there are the values I need to remove from the first row
        for j in range(9):
            if sol[i,j] == 0:
                for key in keys:
                    if num_options(ws[i,j]) >1 and key in ws[i,j]: #otherwise, it will remove the key from its native place
                        ws[i,j]=remove_value_from_vec(ws[i,j],key)

    sol=update_sol(ws,sol)
                        
    return([ws,sol])


def loop_columns(ws,sol):
    #loop for the columns
    for j in range(9):
        keys = shrink_vec(np.unique(sol[:,j])) # there are the values I need to remove from the first row
        for i in range(9):
            if sol[i,j] == 0:
                for key in keys:
                    if num_options(ws[i,j]) >1 and key in ws[i,j]: #otherwise, it will remove the key from its native place
                        ws[i,j]=remove_value_from_vec(ws[i,j],key)    
    
    sol=update_sol(ws,sol)
    return([ws,sol])


def loop_blocks(ws,sol):

    for block in range(9):
        block_keys=shrink_vec(sol[block_limits(block)[0][0]:block_limits(block)[1][0]+1,block_limits(block)[0][1]:block_limits(block)[1][1]+1])
        
        for i in range(block_limits(block)[0][0],block_limits(block)[1][0]+1):
            for j in range(block_limits(block)[0][1],block_limits(block)[1][1]+1):
                if sol[i,j] ==0:
                    for key in block_keys:
                        if key in ws[i,j] and len(shrink_vec(ws[i,j])) >1:
                            ws[i,j]=remove_value_from_vec(ws[i,j],key)
                            if len(shrink_vec(ws[i,j]))==1:
                                print('new key found in',i,j,',block',block,', key',key)
        
    sol=update_sol(ws,sol)
    return([ws,sol])

#=======    
# Aplly firt checking, with all the rows ans columns

#step zero: create the matrix of possible keys
workspace=create_workspace()
print('Step 0 done. N zeros:',np.count_nonzero(workspace==0))

#step 1: apply the initial condition of the sudoku
workspace=initial_conditions(workspace,solution)
print('Step 1. initial_conditions. N zeros:',np.count_nonzero(workspace==0))
# %%


ws=workspace
sol=solution

# loop for the rows
for i in range(3):
    keys = shrink_vec(np.unique(sol[i,:])) # there are the values I need to remove from the first row
    for j in range(3):
        if sol[i,j] == 0:
            for key in keys:
                if num_options(ws[i,j]) >1 and key in ws[i,j]: #otherwise, it will remove the key from its native place
                    #print('changing in', i,j,'.key',key)    
                    #print('ws before')
                    #print(ws[i,j])
                    ws[i,j]=remove_value_from_vec(ws[i,j],key)
                    #print('ws after')
                    #print(ws[i,j])
sol=update_sol(ws,sol)


for j in range(9):
    keys = shrink_vec(np.unique(sol[:,j])) # there are the values I need to remove from the first row
    for i in range(9):
        if sol[i,j] == 0:
            for key in keys:
                if num_options(ws[i,j]) >1 and key in ws[i,j]: #otherwise, it will remove the key from its native place
                    print('changing in', i,j,'.key',key)    
                    #print('ws before')
                    #print(ws[i,j])
                    ws[i,j]=remove_value_from_vec(ws[i,j],key)
                    print('ws after')
                    print(ws[i,j])    

sol=update_sol(ws,sol)




# %%



#step 2: loop over rows, and remove possible values for the keys present in the row
#[workspace,solution]=loop_rows(workspace,solution)
#print('Step 2. Loop over rows. N zeros:',np.count_nonzero(workspace==0))

#step 3: loop over columns, and remove possible values for the keys present in the columns
#[workspace,solution]=loop_columns(workspace,solution)
#print('Step 3. Loop over columns. N zeros:',np.count_nonzero(workspace==0))


#[workspace,solution]=loop_blocks(workspace,solution)
#print('step 4. Loop over blocks. N zeros:',np.count_nonzero(workspace==0))

#loop_over_blocks(workspace,raw)

# %%

#for i in range(9):
#    for j in range(9):
#        if raw[i,j]==0 and solution[i,j] !=0:
#            print('new key in ',i,j, 'as ', solution[i,j])

