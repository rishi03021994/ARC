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

### In each tasks we  have grid and each cell has a number designated to it whihc corresponds to a color for example 0 represents black.

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


'''Task Id ecdecbb3 Transforamtion Description - We have an input grid in which each cell has either one of the three colors: Black(0), Dark orange(2) and 
light blue(8). The input grid is such that there one or more cells with number 2 whose adjacent cells are black(0) as the background is black and there
are some rows or columns in which all cells are light blue(8). In order to obtain the desired output we need to transform the input grid in such a way that 
the cell numbered 2(orange) should replace the cell of the row or column having all cells light blue(8) which ever present in grid and also replace all the cells
with orange which are in between the orange cell(2) and the row or column cell. For example if a cell i,j is orange and row k is complete blue then
all rows between i and k should be changed to orange for column j. But the cells adjacent(including diagonal cells) to the cell in row or column which is replaced by 2, should be
replaced by 8(blue). ANd the rows and columns which are closest to the orange cell in both directions should be used and not all rows and columns.'''
def solve_ecdecbb3(x):
    x_arr = x.copy()
    rows_8 = [i for i, item in enumerate(np.all(x_arr == 8,axis=1)) if item] # Find all row numbers having all values equal to 8

    def transform_x(x_arr,rows_8): # Function to perform the described transformation
        rows_2 = [i for i, item in enumerate(np.any(x_arr == 2, axis=1)) if item] # Get all row numbers where rows have any cell value equal to 2

        for row_2 in rows_2: # Iterate each row number having 2 in any cell
            row = rows_8.copy()
            row.append(row_2) # Append the row number having 2 value with the row numbers list having all values as 8
            row.sort() # Sort the row having row numbers of row number having 2 and row numbers having all 8 in order to get the row numbers of 8 which are closest to row number 2 in both direction
            idx = row.index(row_2) # Find the index of row number 2 in the sorted list
            if (idx == 0): # If the row number of 2 is in the first index this means we need to find a row number of 8 below the row number 2 and that will be in index 1
                idx_8 = [idx+1] # Store the index of row number of 8 closest to row number 2
            elif (idx == len(row)-1):# If the row number of 2 is in the last index this means we need to find a row number of 8 just before the row number 2 and that will be second last index in sorted list
                idx_8 = [idx-1]
            else: # If the row number of 2 is not the first and last index then take the indexes of row numbers 8 index just before  and just after the index of row number 2
                idx_8 = [idx-1,idx+1]
            final_rows_8 = [row[i] for i in idx_8] # Get the rows numbers which are closest to row number 2 in both directions above and below
            for final_row in final_rows_8: #Iterate each row number of 8 to perform transformation
                col_2 = np.where(x_arr[row_2] == 2)[0][0] # Find the column number of the row number having 2 value
                if row_2 < final_row: # If the row number having 2 value is above the row number having all values 8
                    x_arr[final_row, col_2] = 2 # Replace the cell of row number having 8 and the column where row number is 2 with 2
                    x_arr[row_2 + 1:final_row - 1, col_2] = 2 # Replace all cells from row number having 2 till 2 cells above the row number having 8
                    x_arr[final_row + 1, col_2 - 1:col_2 + 2] = 8 # Replace the adjacent cells including diagonal cells of the first cell replaced with 2 of the 2nd last step bfore this
                    x_arr[final_row - 1, col_2 - 1:col_2 + 2] = 8
                else: # If the row number having 2 value is below the row number having all values 8 and the rest is same as in if condition
                    x_arr[final_row, col_2] = 2
                    x_arr[final_row + 1:row_2, col_2] = 2
                    x_arr[final_row + 1, col_2 - 1:col_2 + 2] = 8
                    x_arr[final_row - 1, col_2 - 1:col_2 + 2] = 8
        return x_arr

    if len(rows_8) != 0: # Check if any row is present whose all values are 8
        x_arr = transform_x(x_arr,rows_8) # Pass to function to perform transformation as described
    else: # If a column or columns are present whose all values are 8
        x_arr = x_arr.transpose() # Transpose the grid as the transformation logic is written for row wise operations
        rows_8 = [i for i, item in enumerate(np.all(x_arr == 8, axis=1)) if item] # after grid is transposed find the rows having all values 8(these are columns in original grid)
        x_arr = transform_x(x_arr,rows_8) # Pass to function to perform transformation as described
        x_arr = x_arr.transpose() # Transpose the grid returned after transforamtion to get original required grid format
    x = x_arr
    return x
	'''All the training and test grids are solved correctly for Task ID ecdecbb3'''


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

if __name__ == "__main__":
    # Short Summary on the Python features and libraries you used in the solve * functions and on any commonalities or differences among them
    '''
    The Python library used for each transformation task is numpy and some basic Python features used are simple if else conditions, for loops,
    numpy data slicing, indexing, numpy array transpose, numpy where conditions, list comprehension, list slicing.
    
    Commonality between the task ids c9e6f938 and ecdecbb3 solve functions is background detection which is black in both cases. Difference
    between these two ids are that c9e6f938 is a bi-colored grid and ecdecbb3 is tri-colored grid.
    
    Commonality between the task ids 3631a71a and ecdecbb3 solve functions is that in both tasks input grid needs to be transposed in order to obtain
    desired output.
    
    Commonality between the task ids c1d99e64 and c1d99e64 solve functions is that in both tasks we find a row or column where all cells have a specific
    value in order to obtain a desired output.
    
    Difference between task id c1d99e64 and task ids c9e6f938 and ecdecbb3 is that in task c1d99e64 the input grid is bi-colored and output grid is
    tri-colored while in ids c9e6f938 and ecdecbb3 the number of colors remain same in input and output grid.
    
    Difference between task id c9e6f938 and all other task ids is that in task c9e6f938 the output grid shape is changed while in all other tasks the
    output grid shape is same as input grid shape.
    
    Difference between task id c9e6f938 and all other task ids is that in task c9e6f938 there is not replacement of any cell in input grid in order to
    obtain output grid but in other tasks a cell or cells are being replaced.
    '''
    main()

