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


#view the file
print(raw)

