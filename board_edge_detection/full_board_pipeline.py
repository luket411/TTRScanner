from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt

from show_warped_board import transform_board, transform_size, desired_corners, annotate_fixed_city_points
from line_detection import find_corners, show
np.seterr(all="ignore")

if __name__ == "__main__":
    img_file = "assets/clean_board.jpg"
    img = cv2.imread(img_file, 1)

    show(img)
    source_corners = np.array(sorted(find_corners(img_file, show_output=True), key=lambda x: x.sum()), dtype=np.float32)
    target_corners = np.array(sorted(desired_corners, key=lambda x: x.sum()), dtype=np.float32)
    
    print(source_corners)
    print(target_corners)
    
    warped_board = transform_board(img, source_corners, target_corners, transform_size)
    warped_board = annotate_fixed_city_points(warped_board)
    show(warped_board)