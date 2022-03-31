from sys import path
from os import path as ospath

path.append(ospath.join(ospath.dirname(__file__), ".."))
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from util.get_asset_dirs import dirs, subdirs, get_asset_dirs
from train_detection.Map import Map
from board_handling.feature_detection import find_board

correct_colour = np.array([0,255,0], dtype=np.float32)                  # Green 0
labelled_incorrectly_colour = np.array([255,0,0], dtype=np.float32)     # Red 1
shouldnt_be_labelled_colour = np.array([255,215,0], dtype=np.float32)   # Yellow 2
missed_colour = np.array([0,0,255], dtype=np.float32)                   # Blue 3

colour_converter = [
    correct_colour, labelled_incorrectly_colour, shouldnt_be_labelled_colour, missed_colour
]

def index_to_dir(num, subnum, image_index):
    
    if num == 1:
        return f'assets/1.0 Blank/{image_index}.jpg'
    else:
        dir_colour = dirs[num-2]
        if subnum != 0:
            subdir_split = subdirs[subnum-1].split(",")
            if dir_colour in subdir_split:
                subdir_split.remove(dir_colour)
            subdir = ",".join(subdir_split)
        else:
            subdir = dir_colour
        return f'assets/{num}.{subnum} {dir_colour}-{subdir}/{image_index}.jpg'

    
    

def read_file(results_file):
    results = np.full((7,4,16, 101), 5, dtype=np.uint8)
    with open(results_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, quotechar="'")
        for row in csv_reader:
            file = row[0]
            
            split_file = file.split("/")
            first_index = int(split_file[1][0])
            second_index = int(split_file[1][2])
            third_index = int(split_file[2].split(".")[0])
            
            # print(f"{first_index}.{second_index}.{third_index}")
            
            results[first_index, second_index, third_index] = row[1:]
    return results


        
def display_results(results,*index):
    file = index_to_dir(*index)
    img, _ = find_board("assets/0.0 Cropped/3.png", file)
    
    result = results[index]
    
    print_result_info(result)    
    
    
    map = Map()
    plt.imshow(img)
    for id, connection in enumerate(map.connections):
        connection.plot(use_colour=colour_converter[result[id]])
    plt.show()


def print_result_info(result):
    correct = (result == 0).sum()
    labelled_incorrectly = (result == 1).sum()
    shouldnt_be_labelled = (result == 2).sum()
    missed = (result == 3).sum()
    
    print(f"Connections labelled correctly (Green or no highlight): {correct}")
    print(f"Connections labelled with wrong colour (Red): {labelled_incorrectly}")
    print(f"Connections labelled when no train exists (Yellow): {shouldnt_be_labelled}")
    print(f"Connections not labelled when train exits (Blue): {missed}")

def display_all(results):
    for num, train_col_row in enumerate(results):
        for sub_num, board_col_row in enumerate(train_col_row):
            for image_num, image_data in enumerate(board_col_row):
                if image_data.max() != 5:
                    index = [num, sub_num, image_num]
                    if num == 1:
                        print("=================================")
                        print(index_to_dir(*index))
                        display_results(results, *index)
                        # print_result_info(image_data)    

if __name__ == "__main__":
    results = read_file("run27.03.csv")
    display_results(results, 4,1,6)
    # display_all(results)
    