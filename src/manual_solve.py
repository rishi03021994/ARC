#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

'''Task ID c9e6f938 Transformation Description - We have an input 3x3 grid in the particular task and each cell has either Black(0) or Orange(7) color.
The tranformation which is needed in this task to get desired output is to reshape the matrix to 3x6 and just take the mirror image on Y-axis of the 3x3 matrix to the right
side of the Y-axis and concatenate the mirrored 3x3 matrix with original one which will result in 3x6 matrix.'''
def solve_c9e6f938(x):
    new_xarr = []
    for train_x in x: # Iterate the 3x3 array and take each row of the array
        rev_train_x = train_x[::-1] # For each row get the reversed or the mirrored row
        new_xarr.append(np.concatenate((train_x, rev_train_x), axis=0)) # Concatenate the original row with the reversed row using axis = 0 and append it into new list
    x = np.array(new_xarr) # Convert the list to numpy array
    return x
'''All the training and test grids are solved correctly for Task ID c9e6f938'''

'''Task ID c1d99e64 Transformation Description - We have an input grid and the grid has two different colors either black(0) cell or any ther color.
To tranform the grids in this task we need to find all rows and columns in grid whose all cells are black in color or have 0 in them and then 
for those rows or columns replace all the cells with red(2) color.'''
def solve_c1d99e64(x):
    new_xarr = x.copy()
    rows = np.where(~new_xarr.any(axis=1))[0] # Find all rows in the grid where all cells in the row are having 0 value(black)
    cols = np.where(~new_xarr.any(axis=0))[0] # Find all columns in the grid where all cells in the column are having 0 value(black)
    for col in cols: # Iterate all such columns having all values as 0
        new_xarr[:,col] = 2 # Replace the all cells value with 2 in the particular column
    for row in rows: # Iterate all such rows having all values as 0
        new_xarr[row,:] = 2 # Replace the all cells value with 2 in the particular row
    x = new_xarr
    return x
'''All the training and test grids are solved correctly for Task ID c1d99e64'''

'''Task ID 3631a71a Transformation Description - We have  an input grid which is combination of multiple colors (each cell has a number from 0 to 8).
The issue with the input grid is that there are some blocks of cells having red color(9) which can be thought as broken tiles(9 numbered) on a wall of 
some pattern. The transformation that we need to do to get our desired output is to replace those red block of cells with the most matching pattern in
the grid. We need to check each row having those block of red cells and find a pattern in other rows which is closest to this row(ignoring red cells(9)).
If we find a match then replace the row with the row having closest pattern. If no row match is found look at the columns which match a as done for rows to
find a perfect match. There might be a scenario like in test grid where a row does not match with any row and after transposing a column does not match 
with any column then for a row we can find a column which matches and vice versa'''
def solve_3631a71a(x):
    x_arr = x.copy()
    def replace_rows(x_arr, x_arr_to_check):
        for i, row in enumerate(x_arr): #Iterate each row of the grid to perform transformation
            if 9 not in row: # Check whether the row consists of the 9 value and if not then the row need not be transformed.
                continue
            replacement = None
            #If the row consists of 9 value
            for j, row_to_check in enumerate(x_arr_to_check): # This for loop is for finding perfect matched row for the row having 9 value
                if i != j and can_be_replaced(row, row_to_check): # Check the row that needs to be transformed is the not same row as we are
                    # checking in same input grid and if not then pass the row to be transformed and the row to match to a function where it
                    # checks if the row matches perfectly
                    replacement = row_to_check
            if replacement is None:
                continue
            x_arr[i] = replacement # Once the perfect row is found then replace with the row to be transformed
        return x_arr
    '''All the training and test grids are solved correctly for Task ID 3631a71a'''


    def can_be_replaced(row_with_9s, row_to_check): # A function to check if two rows have same values in same positions apart from 9 for row to be transformed
        for val1, val2 in zip(row_with_9s, row_to_check): # To check values at same position in both array
            if val1 == 9: # If value in row to be transformed is 9 then ignore as we need to check non nine values
                continue
            if val1 != val2: # If any value at a particular position of row to be transformed does not match with the value at that position of the row
                # check for matching then return False and this row cannot be used for replacement
                return False
        return True

    x_arr = replace_rows(x_arr, x_arr) # Find if a row having block of red cells can replaced with any other row in the grid
    x_arr = replace_rows(x_arr, x_arr.transpose()) # Find if a row having block of red cells can replaced with any other column in the grid
    x_arr = replace_rows(x_arr.transpose(), x_arr.transpose()) # Find if a column having block of red cells can replaced with any other column in the grid
    x_arr = x_arr.transpose()  # Transpose the final grid to get the output grid in the correct format
    x = x_arr
    return x
'''All the training and test grids are solved correctly for Task ID 3631a71a'''

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

