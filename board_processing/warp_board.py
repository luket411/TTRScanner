from sys import path
from os import path as ospath
path.append(f'{ospath.dirname(__file__)}/..')
from datasets import blank_board
import numpy as np
import cv2


desired_corners = np.array([[0,0],[0,2000],[3000,2000],[3000,0]], dtype='float32')
transform_size = (3000,2000)

def transform_board(board, original, desired, transform_size):
    original = np.array(original, dtype='float32')
    desired = np.array(desired, dtype='float32')
    M = cv2.getPerspectiveTransform(original, desired)
    transformed = cv2.warpPerspective(board, M, transform_size)    
    return transformed


def main(clean_board_file = "assets\\1.0 Blank\\PXL_20220209_145822567.jpg"):
    corners = [[164,61],[58,1908],[2907,1932],[2859,125]]
    board = cv2.imread(clean_board_file, 1)
    transformed = transform_board(board, corners, desired_corners, transform_size)
    return transformed