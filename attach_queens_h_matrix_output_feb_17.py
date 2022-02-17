# aut:-chaitanyantr.github.io
# 8 queens problem layout
# each col have only 1 queen
# read csv file of map.txt
import time

from numpy import genfromtxt
import numpy as np
import copy
import math

grid_array = genfromtxt('map.txt', delimiter=',')

def queen_loc_in_grid_weight_in_col(grid, col_num):
    'given grid and col_num returns queen location(greater than 1 in 2darray) in that column and value of index'
    row_count = 0
    grid_col = np.array(grid).T.tolist()
    for _ in grid_col[col_num]:
        row_count += 1
        if _ >= 1:
            return [row_count - 1, col_num], grid[row_count - 1, col_num]

def how_many_im_attaching(grid, row, col):
    '''for queen horizontal,vertical,diagonal elements are attaching area, it have both back and front directions'''

    def horizontal_both_directions(row, col):

        hor_attach_count = 0
        grid_array_row = np.delete(grid[row], col)

        for each_element_in_row in grid_array_row:
            if int(each_element_in_row) >= 1:
                hor_attach_count += 1

        # print('hor_count',hor_attach_count)

        return hor_attach_count

    def vertical_both_directions(row, col):

        grid_array_col = np.array(grid).T.tolist()
        grid_array_col = np.delete(grid_array_col[col], row)

        ver_attach_count = 0

        for each_element_in_col in grid_array_col:
            if int(each_element_in_col) >= 1:
                ver_attach_count += 1

        # print('vertical_Count',ver_attach_count)

        return ver_attach_count

    def diagonal_both_directions(row, col):

        index = [row, col]

        diagonals = [[[index[0] + i, index[1] + i] for i in range(1, 8)],
                     [[index[0] + i, index[1] - i] for i in range(1, 8)],
                     [[index[0] - i, index[1] + i] for i in range(1, 8)],
                     [[index[0] - i, index[1] - i] for i in range(1, 8)]]

        diagonal_indexes = []
        diagonal_count = 0

        for i in range(4):
            for x, y in diagonals[i]:
                if 0 <= x < 8 and 0 <= y < 8:
                    diagonal_indexes.append([x, y])

        for d_row, d_col in diagonal_indexes:
            if grid_array[d_row][d_col] > 0:
                diagonal_count += 1

        # print('dia_Count',diagonal_count)

        return diagonal_count

    hor_count = horizontal_both_directions(row, col)
    ver_count = vertical_both_directions(row, col)
    dia_count = diagonal_both_directions(row, col)

    all_attachs = hor_count + ver_count + dia_count

    return hor_count, ver_count, dia_count, all_attachs

def grid_h_matrix(grid):
    'give a 2d grid return the h_matrix based on heristic equation'
    h_matrix = np.zeros((8, 8))

    queen_loc_in_grid_weight_in_col(grid)

    for r in range(8):

        for c in range(8):

            h, v, t, no_of_attachs = how_many_im_attaching(grid, r, c)

            heristic = 100*no_of_attachs

            h_matrix[r][c] = heristic

    return h_matrix

def min_index_in_grid(grid):
    'returns a single list index of min value in a grid/2d array'
    min_index = np.unravel_index(np.argmin(grid, axis=None), grid.shape)
    return min_index

def min_value_in_grid(grid):
    'returns only min value of a grid/2d array'
    min_value = np.min(grid)
    return min_value

def distance(ref_grid,curr_grid,col):
    '''#find queen loc in ref_grid based on col
    #find queen loc in current grid based on col'''

    ref_grid_queen_pos,_ = queen_loc_in_grid_weight_in_col(ref_grid,col)
    cur_grid_queen_pos,_ = queen_loc_in_grid_weight_in_col(curr_grid,col)

    # print('ref_loc',ref_grid_queen_pos[0][0])
    # print('cur_loc',cur_grid_queen_pos[0][0])
    # print((cur_grid_queen_pos[0][0]-ref_grid_queen_pos[0][0])**2)

    dist = math.sqrt((cur_grid_queen_pos[0]-ref_grid_queen_pos[0])**2 + (cur_grid_queen_pos[1]-ref_grid_queen_pos[1])**2)
    print(dist)

    return dist

def move_queen(ref_grid,row,col):
    '''find queen in the grid w.r.t to col '''

    queen_loc,queen_weight = queen_loc_in_grid_weight_in_col(ref_grid,col)

    # print(queen_loc,queen_weight)

    ref_grid[queen_loc[0]][queen_loc[1]] = 0

    # print(ref_grid)

    ref_grid[row,col] = queen_weight

    # print(ref_grid)
    return ref_grid

def h_grid(ref_grid):
    # print('parm',ref_grid)
    # print(ref_grid)

    raw_grid_queen_moving = copy.deepcopy(ref_grid)

    raw_grid = copy.deepcopy(ref_grid)

    h_grid = np.zeros((8, 8))

    for col in range(8):

        for row in range(8):
            print('curent loop')
            print('row,col', row, col)

            queen_moving_grid = move_queen(raw_grid_queen_moving,row,col)

            # print('---input grid to how many attachs------')
            # print(queen_moving_grid)
            # print('---------------------------------------')
            # ref_grid[row][col] = 0

            if row == 7:
                '''keep the queen as per the ref_grid once quuen cover all positoins in that col'''
                print('input to queen loc')
                print(raw_grid)
                print('-----------------')
                queen_raw_loc,queen_raw_we = queen_loc_in_grid_weight_in_col(raw_grid,col)
                print('------row becomes 7,so replace queen to original position')
                print('what is queen pos in original grid',queen_raw_loc,queen_raw_we)
                # ref_grid[queen_raw_loc[0],queen_raw_loc[1]] = queen_raw_we
                queen_moving_grid[queen_raw_loc[0],queen_raw_loc[1]] = queen_raw_we
                print('-----after changing to original position grid is----------')
                print(queen_moving_grid)
                print('-------------------------')

                if queen_raw_loc[0] != 7:
                    # ref_grid[row],[col] == 0
                    print('become zero')
                    queen_moving_grid[[row],[col]] = 0

                    print('after zero',queen_moving_grid)

            # print('----ref grid-----------')
            # print(ref_grid)
            # print('-----------------------')
            # time.sleep(1)
            print('queen moving grid')
            print(queen_moving_grid)
            print('-----------------')
            _,_,_,total_attachs = how_many_im_attaching(queen_moving_grid,row, col)
            print('attchs',total_attachs)

            #patch work because of one iteratinoal problem (quuen moving grid becomes 0 and replaced queen to original pos, after doing this qurying total attches with row,col assigns self queen attach in 7th row )

            if row <=6:
                h_grid[row][col] = total_attachs
            elif row == 7 and total_attachs > 1:
                h_grid[row][col] = total_attachs - 1
            elif row ==7 and total_attachs == 0:
                h_grid[row][col] = 0

            print('----h_grid----')
            print(h_grid)
            print('--------------')

            # ref_grid[loc[0], loc[1]] = we

    # print(h_grid)

h_grid(grid_array)
# _, _, _, a = how_many_im_attaching(np.array([[0,0,0,0,5,0,0,3],[0,2,0,0,0,0,0,0],[0,0,0,4,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,3,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[1,0,0,0,0,0,0,0]]), 0, 7)
# print(a)
























# def max_value_in_grid(grid):
#     'returns max value index of a grid/2d array'
#     max_value = np.max(grid)
#     return max_value


#
# def min_loc_in_grid_col(grid,col):
#     queen_loc_min_col = queen_loc_in_col(grid,col)
#     return queen_loc_min_col
#
#
# def move_queen_to_min(h_grid,grid,col_wise):
#
#     # queen_pos, queen_weight = queen_loc_in_col(grid, col_wise)
#     # h_grid[queen_pos[0], queen_pos[1]] = 999
#
#     min_loc = min_in_h_matrix(h_grid)
#     print('min_location_in_h_grid',min_loc)
#     print('loc of queen in col',min_loc[1])
#
#     queen_pos_give_col,queen_pos_weight_give_col = min_loc_in_grid_col(grid,min_loc[1])
#
#     print('queen_pos',queen_pos_give_col)
#     print('queen_weight',queen_pos_weight_give_col)
#
#     print('making coresponding queen position 999 in h matrix',queen_pos_give_col)
#     # h_grid[queen_pos[0], queen_pos[1]] = 999
#     print(h_grid)
#
#     # grid[queen_pos[0],queen_pos[1]] = 999
#
#     #-----------------move queen to new location-----------#
#     print('moving queen to min_loc in grid',min_loc)
#     # print('what is min_loc',min_loc)
#     # print()
#
#     grid[min_loc[0],min_loc[1]] = queen_pos_weight_give_col
#     grid[queen_pos_give_col[0], queen_pos_give_col[1]] = 0
#     print(grid)
#
#     return h_grid,grid
#
#
#
# for i in range(7):
#     h_matrix = grid_h_matrix(grid_array)
#     h_value = min_in_h_matrix(h_matrix)
#     print(h_value)
#     for j in range(8):
#         queen_pos, queen_weight = queen_loc_in_col(grid_array, j)
#         h_matrix[queen_pos[0], queen_pos[1]] = 999
#
#
#     h_grid_new, grid_new = move_queen_to_min(h_matrix,grid_array,i)
#     grid_array = grid_new
