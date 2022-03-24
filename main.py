from os import listdir, makedirs, path as ospath
import cv2
import numpy as np
import matplotlib.pyplot as plt

from board_handling.feature_detection import find_board
from train_detection.BoardSegment import BoardSegment

from train_detection.Connection import Connection
from train_detection.Map import Map
from train_detection.data_readers import read_connections, COLOURS

from util.timer import timer, timer_context
from util.constants import BASE_BACKGROUND_COLOUR, COLOURS, INVERSE_COLOURS
from util.Counter import Counter

from datasets.dataset import ImageFileDataset

np.set_printoptions(suppress=True)

# Used to assess full boards.
# Displays a board with each connection highlighted by the colour it thinks is there
def run_no_answers(target_file):
    v = 3
    base_file = f"assets/0.0 Cropped/{v}.png"
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    
    board, _ = find_board(base_file, target_file)
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)

    results = map.process_multicore(board)

    plt.imshow(board)
    for [idx, hasTrain, col] in results:
        connection = map.connections[idx]
        if hasTrain:
            connection.plot(use_colour=np.array(INVERSE_COLOURS[col], dtype=np.float32))
    plt.show()



def main(target_file, train_colour=None, show=True):
    print(target_file)    
    v = 3
    base_file = f"assets/0.0 Cropped/{v}.png"
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    target = int(target_file.split("/")[-1].split(".")[0])

    board, base_board = find_board(base_file, target_file)
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)
    blank_background = np.full(board.shape, BASE_BACKGROUND_COLOUR)


    answer_index = "/".join(target_file.split("/")[:-1])
    answers = read_connections(answer_index)[target]

    # segment: BoardSegment
    # segment = map.connections[0].segments[0]
    # print(segment.containsCarriage(board).getWinner())
    # segment.plot(image=board, show=True)

    # connection: Connection = map.connections[12]
    # print(connection.hasTrain(board))
    # connection.plot(image=board, show=True, image_focus=True)

    asses_board(map, board, answers, show)
    # guess_colours(map, board)

# Uses the provided answers to mark into different categories of how the program performed
def asses_board(map: Map, board, answers, show=True):
    plt.imshow(board)
    results = map.process_multicore(board)
    
    correct = []
    shouldnt_be_labelled = []
    labelled_incorrectly = []
    missed = []
    
    correct_colour = np.array([0,255,0], dtype=np.float32)
    labelled_incorrectly_colour = np.array([255,0,0], dtype=np.float32)
    shouldnt_be_labelled_colour = np.array([255,215,0], dtype=np.float32)
    missed_colour = np.array([0,0,255], dtype=np.float32)
    
    for [idx, islabelled, col] in results:
        connection = map.connections[idx]

        hasTrain = idx in answers.keys()
        right_colour = None if not hasTrain else answers[idx]
        labelled_right_colour = islabelled and right_colour == col
        labelled_wrong_colour = islabelled and not (right_colour == col)
        
        if hasTrain and labelled_right_colour:
            correct.append(connection)
            connection.plot(use_colour=correct_colour)
        elif not hasTrain and not islabelled:
            correct.append(connection)
        elif hasTrain and labelled_wrong_colour:
            labelled_incorrectly.append(connection)
        elif not hasTrain and islabelled:
            shouldnt_be_labelled.append(connection)
        elif hasTrain and not islabelled:
            missed.append(connection)

    for connection in labelled_incorrectly:
        connection.plot(use_colour=labelled_incorrectly_colour)
        
    for connection in shouldnt_be_labelled:
        connection.plot(use_colour=shouldnt_be_labelled_colour)
    
    for connection in missed:
        connection.plot(use_colour=missed_colour)
    
    print(f"Connections labelled correctly (Green or no highlight): {len(correct)}")
    print(f"Connections labelled with wrong colour (Red): {len(labelled_incorrectly)}")
    print(f"Connections labelled when no train exists (Yellow): {len(shouldnt_be_labelled)}")
    print(f"Connections not labelled when train exits (Blue): {len(missed)}")
    if show:
        plt.show()

# Displays a map with the current boards estimate for each connection.
def guess_colours(map: Map, board):
    plt.imshow(board)
    results = map.process_multicore_results(board)
    correct = Counter()
    incorrect = Counter()
    for [idx, counter] in results:
        connection = map.connections[idx]
        if counter.getWinner() == COLOURS[connection.base_colour]:
            correct.addVote(COLOURS[connection.base_colour])
        else:
            incorrect.addVote(COLOURS[connection.base_colour])
        col = np.array(INVERSE_COLOURS[counter.getWinner()], dtype=np.float32)
        connection.plot(use_colour=col)
            
    correct.printBreakdown()
    
    incorrect.printBreakdown()
    plt.show()

# Runs main on all assets
def run_all():
    for main_label in range(1,7):
        for sub_label in range(0,4):
            index = float(f"{main_label}.{sub_label}")
            dataset = ImageFileDataset(index)

            for asset in dataset:
                print(f"Input Image: {asset}")
                main(asset)
                print("========================")

            if index == 1.0:
                break

def test_blank_board(all=False):
    dataset = ImageFileDataset(1.0)
    for asset in dataset:
        main(asset, show=False)
        if not all:
            break

if __name__ == "__main__":

    # test_blank_board(all=True)

    dataset = ImageFileDataset(1.0)
    main(dataset.getImageFileByKey(2))
    # run_no_answers(dataset.getAsset())
    # main(dataset.getAsset())
