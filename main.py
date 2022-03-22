import cv2
import numpy as np
import matplotlib.pyplot as plt

from board_handling.feature_detection import find_board
from train_detection.Connection import Connection
from train_detection.Map import Map
from util.timer import timer
from util.constants import BASE_BACKGROUND_COLOUR

np.set_printoptions(suppress=True)

def main(target_file):
    v = 3
    base_file = f"assets/0.0 Cropped/{v}.png"
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    
    
    board, base_board = find_board(base_file, target_file)
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)
    
    return map, board

        
if __name__ == "__main__":
    map, board = main("assets/4.3 Green-White,Red,Pink/3.jpg")
    
    results = map.process_multicore(board)