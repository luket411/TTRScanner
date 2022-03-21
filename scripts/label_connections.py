from sys import path
from os import path as ospath, listdir
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt

from util.get_asset_dirs import get_asset_dirs, colour_combos
from board_handling.feature_detection import find_board
from util.constants import BASE_BACKGROUND_COLOUR
from train_detection.data_readers import read_connections
from train_detection.Connection import Connection
from train_detection.Map import Map
from scripts.validate_connections import convertor, dir_to_run

def main():
    v = 3
    base_file = f"assets/0.0 Cropped/{v}.png"

    for asset_dir in (get_asset_dirs()[convertor[dir_to_run]:]):
        files = [file for file in listdir(asset_dir) if file[-4:] == '.jpg' or file[-4:] == '.png']
        for file_name in files:
            target_file = f"{asset_dir}/{file_name}"
            print(target_file)
            board, _ = find_board(base_file, target_file)
            map = Map()

            target_file_split = target_file.split('/')
            main_num = target_file_split[1][:3]
            sub_num = target_file_split[2].split('.')[0]
            plt.title(f"{main_num}/{sub_num}")
            plt.imshow(board)

            for conn in map.connections:
                if conn.base_colour in colour_combos[main_num]:
                    conn.plot(label=True)
            plt.show()
        break


if __name__ == "__main__":
    main()