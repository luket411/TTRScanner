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

from datasets.dataset import ImageFileDataset

np.set_printoptions(suppress=True)

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



def main(target_file, train_colour=None):
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

    # hsv_board = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
    # hues = hsv_board[:,:,0]
    # sats = hsv_board[:,:,1]
    # vals = hsv_board[:,:,2]
    # print(f"hues, min:{hues.min()}, max:{hues.max()}, median:{np.median(hues)}, average:{np.average(hues)}")
    # print(f"sats, min:{sats.min()}, max:{sats.max()}, median:{np.median(sats)}, average:{np.average(sats)}")
    # print(f"vals, min:{vals.min()}, max:{vals.max()}, median:{np.median(vals)}, average:{np.average(vals)}")
    # return

    # segment: BoardSegment
    # segment = map.connections[37].segments[0]
    # segment.plot(image=board, show=True)
    # segment.containsCarriage(board)

    # connection: Connection = map.connections[12]
    # print(connection.hasTrain(board))
    # connection.plot(image=board, show=True, image_focus=True)

    # handle_connection_results(map, board, answers, train_colour)
    asses_board(map, board, answers)

def asses_board(map: Map, board, answers):
    plt.imshow(board)
    results = map.process_multicore(board)
    incorrect = []
    correct = []
    for [idx, hasTrain, col] in results:
        connection = map.connections[idx]
        if (hasTrain and idx in answers) or (idx not in answers and not hasTrain):
            correct.append(connection)
            if hasTrain:
                connection.plot(use_colour=np.array([0,255,0], dtype=np.float32))
        else:
            incorrect.append(connection)

        if hasTrain:
            print(f"Connection: {str(connection)} has a Train of colour: {col}")

    for incorrect_connection in incorrect:
        incorrect_connection.plot(use_colour=np.array([255,0,0], dtype=np.float32))
        print(f"Connection: {incorrect_connection}({incorrect_connection.id}) was mislabelled")
    
    print(f"Connections marked correctly: {len(correct)}")
    print(f"Connections marked incorrectly: {len(incorrect)}")
    plt.show()

def handle_connection_results(map, board, answers, train_colour):
    results = map.process_multicore_results(board)
    incorrects = []
    
    successes, total = 0, 0
    for [connection_idx, connection_result] in results:
        connection = map.connections[connection_idx]
        connection_colour = COLOURS[connection.base_colour] if connection_idx not in answers else train_colour
        if connection_result.getWinner() == connection_colour:
            successes += 1
        else:
            incorrects.append([connection, connection_result])
        total += 1
        # print(f"Connection: {str(connection)}({connection.id}), Base Colour: {COLOURS[connection.base_colour]}, Predicted Colour: {connection_result.getWinner()}, Confidence: {round(connection_result.getWinningPercentage(connection.size)*100)}%")

    print(f"{successes}/{total}")
    
    for [connection, result] in incorrects:
        print(f"Connection: {str(connection)}({connection.id}), Base Colour: {COLOURS[connection.base_colour]}, Predicted Colour: {result.getWinner()}, Confidence: {round(result.getWinningPercentage(connection.size)*100)}%")

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

if __name__ == "__main__":

    dataset = ImageFileDataset(4.2)
    # run_no_answers(dataset.getAsset())
    main(dataset.getAsset())
