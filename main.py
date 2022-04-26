from os import path as ospath
import cv2
import numpy as np
import matplotlib.pyplot as plt

from board_handling.feature_detection import find_board
from train_detection.BoardSegment import BoardSegment

from train_detection.Connection import Connection
from train_detection.Map import Map
from train_detection.data_readers import read_connections, COLOURS

from util.timer import timer
from util.constants import BASE_BACKGROUND_COLOUR, COLOURS, CONNECTION_COLOURS, INVERSE_COLOURS, INVERSE_CONNECTION_COLOURS
from util.Counter import Counter

from datasets.dataset import ImageFileDataset, index_to_dir, get_all_of_piece_colour, get_all_of_tile_colour, random_dataset, get_file_code

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

def ttr_scanner(target_file, show=True):
    retVal = ttr_scanner_with_answers(target_file, show)
    return retVal[0:2]

def ttr_scanner_with_answers(target_file, show):
    print(f"Running ttr_scanner on: {target_file}")
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

    results = asses_board(map, board, answers, show)
    return [target_file, results, answers]

def test_connection(target_file, connection_number):
    v = 3
    base_file = f"assets/0.0 Cropped/{v}.png"
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"

    board, _ = find_board(base_file, target_file)
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)
    connection: Connection = map.connections[connection_number]
    print(connection.hasTrainDebug(board))
    connection.plot(image=board, show=True, image_focus=True, use_avg_colour=True)

# Uses the provided answers to mark into different categories of how the program performed
def asses_board(map: Map, board, answers, show=True):
    plt.imshow(board)
    
    # Big Step!
    results = map.process_multicore(board)
    
    correct_hasTrain = []       # Green 0
    correct_noTrain = []        # No colour 1
    labelled_incorrectly = []   # Red 2
    shouldnt_be_labelled = []   # Yellow 3
    missed = []                 # Blue 4
    
    correct_colour = np.array([0,255,0], dtype=np.float32)                  # Green 0
    labelled_incorrectly_colour = np.array([255,0,0], dtype=np.float32)     # Red 2
    shouldnt_be_labelled_colour = np.array([255,215,0], dtype=np.float32)   # Yellow 3
    missed_colour = np.array([0,0,255], dtype=np.float32)                   # Blue 4
    
    retVal = [[], [], [], [], []]
    
    
    for [idx, islabelled, col] in results:
        connection = map.connections[idx]

        hasTrain = idx in answers.keys()
        right_colour = None if not hasTrain else answers[idx]
        labelled_right_colour = islabelled and right_colour == col
        labelled_wrong_colour = islabelled and not (right_colour == col)
        
        if hasTrain and labelled_right_colour:
            correct_hasTrain.append(connection)
            connection.plot(use_colour=correct_colour)
            retVal[0].append(connection.id)
        elif not hasTrain and not islabelled:
            correct_noTrain.append(connection)
            retVal[1].append(connection.id)
        elif hasTrain and labelled_wrong_colour:
            labelled_incorrectly.append(connection)
        elif not hasTrain and islabelled:
            shouldnt_be_labelled.append(connection)
        elif hasTrain and not islabelled:
            missed.append(connection)

    for connection in labelled_incorrectly:
        retVal[2].append(connection.id)
        connection.plot(use_colour=labelled_incorrectly_colour)
        
    for connection in shouldnt_be_labelled:
        retVal[3].append(connection.id)
        connection.plot(use_colour=shouldnt_be_labelled_colour)
    
    for connection in missed:
        retVal[4].append(connection.id)
        connection.plot(use_colour=missed_colour)
    
    if len(correct_hasTrain):
        print(f"Connections labelled correctly (Green): {len(correct_hasTrain)}")
    if len(correct_noTrain):
        print(f"Empty connections labelled correctly (No Colour): {len(correct_noTrain)}")
    if len(labelled_incorrectly):
        print(f"Connections labelled with wrong colour (Red): {len(labelled_incorrectly)}")
    if len(shouldnt_be_labelled):
        print(f"Connections labelled when no train exists (Yellow): {len(shouldnt_be_labelled)}")
    if len(missed):
        print(f"Connections not labelled when train exits (Blue): {len(missed)}")

    if show:
        plt.show()
    return retVal

# Displays a map with the current boards estimate for each connection.
def guess_colours(board_file):
    
    board, _ = find_board("assets/0.0 Cropped/3.png", board_file)
    map = Map()
    
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
    print("\n")
    incorrect.printBreakdown()
    plt.show()

# Runs main on all assets
def run_all(csv_file, dir):
    if ospath.exists(csv_file) and ospath.getsize(csv_file) != 0:
        name = csv_file[:-4]
        counter = 1
        while ospath.exists(csv_file):
            csv_file = f"{name}({counter}).csv"
            counter += 1
        
    print(f"Writing to {csv_file}")
    
    returns = []
    for main_label in range(1,7):
        for sub_label in range(0,4):
            index = float(f"{main_label}.{sub_label}")
            dataset = ImageFileDataset(index)
            for asset in dataset:
                returns.append(result := ttr_scanner(asset, show=False))
                result_to_csv(result, dir)
                print("========================")

            if index == 1.0:
                break
    
    print_results(returns, csv_file)

def result_to_csv(result, dir, connection_results=True):
    file = result[0]
    file_code = get_file_code(file)
    if connection_results:
        report = print_connection_col_breakdown(result, print_report=False)
    else:
        report = print_train_col_breakdown(result, print_report=False)
    
    out_file = ospath.join(dir,f"{file_code}.csv")
    
    with open(out_file, "a") as file:
        file.write(report)
    

def print_results(result_set, csv_file):
    output_String = ""
    for [file, results] in result_set:
        output_String += f"'{file}',"
        connection_values = np.empty(101, np.uint8)
        for result, connection_ids in enumerate(results):
            for connection_id in connection_ids:
                connection_values[connection_id] = result
        output_String += ",".join(connection_values.astype(str))
        output_String += "\n"
    print(output_String)

    with open(csv_file, "w") as file:
        file.write(output_String)

    return output_String
        

def test_blank_board(all=False, show=False):
    dataset = ImageFileDataset(1.0)
    for asset in dataset:
        ttr_scanner(asset, show)
        if not all:
            break

def test_all_piece_col(col, dir, all=True, show=False):
    assets = get_all_of_piece_colour(col)
    results = []
    for asset in assets:
        result = ttr_scanner_with_answers(asset, show)
        file = result[0]
        result_set = result[1]
        covered_connections = result[2].keys()
        
        placed_trains = [
            [],
            [],
            [],
            [],
            []
        ]
        
        for category, result_selection in enumerate(result_set):
            for res in result_selection:
                if res in covered_connections:
                    placed_trains[category].append(res)
        
        formatted_result = [
            file,                               # Sample Image
            placed_trains,                      # Format of files
            file.split(" ")[1].split("-")[0]    # Colour
        ]
        
        results.append(formatted_result)
        result_to_csv(formatted_result, dir, False)
        if not all:
            break
    return results

def test_all_tile_col(col, all=True, show=False):
    assets = get_all_of_tile_colour(col)
    for asset in assets:
        ttr_scanner(asset, show)
        if not all:
            break

# Accepts a results list generated by ttr_scanner() and prints a breakdown of results by tile colour
def print_connection_col_breakdown(results, print_report=True):
    file = results[0]
    
    seperated_results = [
        [[],[],[],[],[]],# "Red"],
        [[],[],[],[],[]],# "Green"],
        [[],[],[],[],[]],# "Pink"],
        [[],[],[],[],[]],# "Orange"],
        [[],[],[],[],[]],# "Black"],
        [[],[],[],[],[]],# "Gray"],
        [[],[],[],[],[]],# "Blue"],
        [[],[],[],[],[]],# "Yellow"],
        [[],[],[],[],[]],# "White"],
    ]
    colour_indexes = list(COLOURS.keys())
    display_colour_indexes = list(INVERSE_COLOURS.keys())
    
    for category, categorised_results in enumerate(results[1]):
        for result in categorised_results:
            connection_colour = CONNECTION_COLOURS[result]
            colour_index = colour_indexes.index(connection_colour)
            
            seperated_results[colour_index][category].append(result)

    out_string = f"\"{file}\",correctly_labelled,correctly_missed,labelled_wrong,shouldnt_be_labelled,missed\n"
    for colour_idx, categorised_results in enumerate(seperated_results):
        categorised_size = ",".join([str(len(results)) for results in categorised_results])
        out_string += f"{display_colour_indexes[colour_idx]},{categorised_size}\n"
     
    if print_report:
        print(out_string)
    return(out_string)


# Accepts a results list generated by ttr_scanner() and prints a breakdown of results by train colour
def print_train_col_breakdown(results, print_report=True):
    file = results[0]
    colour = results[2]
    
    seperated_results = [
        [[],[],[],[],[]],# "Red"],
        [[],[],[],[],[]],# "Green"],
        [[],[],[],[],[]],# "Pink"],
        [[],[],[],[],[]],# "Orange"],
        [[],[],[],[],[]],# "Black"],
        [[],[],[],[],[]],# "Gray"],
        [[],[],[],[],[]],# "Blue"],
        [[],[],[],[],[]],# "Yellow"],
        [[],[],[],[],[]],# "White"],
    ]
    colour_indexes = list(COLOURS.keys())
    display_colour_indexes = list(INVERSE_COLOURS.keys())
    
    for category, categorised_results in enumerate(results[1]):
        for result in categorised_results:
            connection_colour = CONNECTION_COLOURS[result]
            colour_index = colour_indexes.index(connection_colour)
            
            seperated_results[colour_index][category].append(result)

    out_string = f"\"{colour}:{file}\",correctly_labelled,correctly_missed,labelled_wrong,shouldnt_be_labelled,missed\n"
    for colour_idx, categorised_results in enumerate(seperated_results):
        categorised_size = ",".join([str(len(results)) for results in categorised_results])
        out_string += f"{display_colour_indexes[colour_idx]},{categorised_size}\n"
     
    if print_report:
        print(out_string)
    return(out_string)



if __name__ == "__main__":

    # run_all("run31.03.csv", "runs/21.04")

    # test_blank_board(all=True)
    # test_blank_board(show=True)
    
    # test_all_piece_col("green", all=True, show=False)

    # asset = index_to_dir(4,0,1)
    # guess_colours(asset)
    # print(res := ttr_scanner(asset))
    # print_connection_col_breakdown(res)
    
    dir = "runs/26.04"
    for col in ["Red","Blue","Green","Yellow","Black"]:
        test_all_piece_col(col, dir)
