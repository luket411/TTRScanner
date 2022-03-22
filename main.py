from os import listdir, path as ospath
import cv2
import numpy as np
import matplotlib.pyplot as plt

from board_handling.feature_detection import find_board

from train_detection.Connection import Connection
from train_detection.Map import Map
from train_detection.data_readers import read_connections, COLOURS

from util.timer import timer
from util.constants import BASE_BACKGROUND_COLOUR

np.set_printoptions(suppress=True)

def main(target_file):
    v = 3
    base_file = f"assets/0.0 Cropped/{v}.png"
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    answers_file = "/".join(target_file.split("/")[:-1])
    target = int(target_file.split("/")[-1][0])
    
    board, base_board = find_board(base_file, target_file)
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)
    blank_background = np.full(board.shape, BASE_BACKGROUND_COLOUR)
    answers = read_connections(answers_file)[target]

    connection: Connection
    target_file_split = target_file.split('/')
    main_num = target_file_split[1][:3]
    sub_num = target_file_split[2][0]
    plt.title(f"{main_num}/{sub_num}")
    map.plot(show=True, image=board, label=True)

    # map.plot(show=True, image=blank_background)
    # map.plot(show=True, image=board)

    # results = run_multicore(map, board)

    # plot_seperate_results(results, map, answers)

    # connection: Connection = map.connections[0]
    # connection.hasTrain(board)


def plot_all_results(results, map, answers):
    sorted_results = sorted(results, key=lambda x:x[1])
    for pos, [idx, result] in enumerate(sorted_results):
        args = dict(
            c=map.connections[int(idx)].base_colour
        )
        
        if idx in answers:
            args['marker'] ='x'

        plt.scatter([pos], [result], **args)
    
    plt.show()


def plot_seperate_results(results, map, answers):
    sorted_results = sorted(results, key=lambda x:x[1])

    fig, subplots = plt.subplots(10, figsize=(12, 6))
    for pos, [idx, result] in enumerate(sorted_results):
        colour = map.connections[int(idx)].base_colour
        params = dict(
            c=colour
        )

        if idx in answers:
            params['marker'] = 'x'
        
        colour_index = COLOURS.index(colour)

        # subplots[(colour_index//5)-1, (colour_index%5)-1].scatter([pos], [result], **params)
        subplots[colour_index].scatter([pos], [result], **params)

    plt.show()
        
if __name__ == "__main__":
    map, board = main("assets/4.3 Green-White,Red,Pink/3.jpg")
    
    results = map.process_multicore(board)
