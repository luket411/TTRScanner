from sys import path
from os import getcwd, path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import numpy as np
import cv2

desired_corners = np.array([[0,0],
                [0,2000],
                [3000,2000],
                [3000,0]], dtype='float32')
transform_size = (3000, 2000)

def transform_board(board, original, desired=desired_corners, transform_size=transform_size):
    homography = cv2.getPerspectiveTransform(original, desired)
    transformed = cv2.warpPerspective(board, homography, transform_size)
    return transformed

if __name__ == "__main__":
    print(desired_corners.shape)