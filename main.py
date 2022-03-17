import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent import futures

from board_handling.feature_detection import find_board
from train_detection.Connection import Connection
from train_detection.Map import Map
from util.timer import timer

def main(target_file):
    v = 11
    base_file = f"assets/0.0 Cropped/{v}.png"
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    
    
    board, base_board = find_board(base_file, target_file)
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)
    
    x = run_multicore(map, board)
    print(x)
    # run_singlecore(map, board)

@timer
def run_multicore(map, board):
    results = []
    with futures.ProcessPoolExecutor() as executor:
        processes = [executor.submit(connection.hasTrain, board) for connection in map.connections]
        for process in processes:
            results.append(process.result())
    return results

@timer
def run_singlecore(map, board):
    results = []
    for connection in map.connections:
        results.append(connection.hasTrain(board))
    return results


if __name__ == "__main__":
    main("assets/0.0 Cropped/11.png")
    # main("assets/2.2 Red-Blue,Black,Orange/PXL_20220209_151954858.jpg")
    # main("assets/2.1 Red-Yellow,Green,Gray/PXL_20220209_151741865.jpg")