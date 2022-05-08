from sys import path
from os import makedirs, path as ospath

path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

from util.timer import timer_context
from train_detection.Map import Map
from train_detection.data_readers import read_connections
from train_detection.BoardSegment import BoardSegment
from util.geometry import Point
from board_handling.feature_detection import find_board
from datasets.dataset import get_all_images, get_file_code, get_asset_dirs, index_to_dir

base_board_version = 3
base_board = f"assets/0.0 Cropped/{base_board_version}.png"

def main():
    map = Map()
    for asset in get_all_images():

        board = find_board(base_board, asset)[0]

        file_code = get_file_code(asset)

        dir = ospath.join("CNN_training", file_code)

        with timer_context(f"pixel finding for: {file_code}"):
            for [segment_image, id] in map.find_segments_in_image(board):
                filename = ospath.join(dir, f"{id}.png")
                cv2.imwrite(filename, cv2.cvtColor(segment_image, 4))
    
        break
    
        print(f"Finished: {dir}")

def generate_labels():
    map = Map()
    
    labels_string = ""
    
    for dir in get_asset_dirs():
        labels = read_connections(dir)
        for image_index in labels.keys():
            file_code = get_file_code(f"{dir}/{image_index}.jpg")
            labels_for_file = labels[image_index]
            labels_string += f"{file_code}"
            
            segment_data = np.full(300, "None", dtype='<U100')
            
            
            for connection_id in labels_for_file.keys():
                for segment_id in map.get_segments_made_from_connection(connection_id):
                    segment_data[segment_id-1] = labels_for_file[connection_id]
                    
            for colour in segment_data:
                labels_string += f",{colour}"
                
            labels_string += "\n"
    
    
    with open("CNN_training\labels.csv", "w") as file:
        file.write(labels_string)
    
    print(labels_string)

def read_csv(csv_file):
    with open(csv_file, "r") as csv_file:
        reader = csv.reader(csv_file)
        return [line for line in reader]

def assess_labels(csv_file):
    main()
    # labels = np.array(read_csv(csv_file))
    # print(labels.shape)
    
                
    

if __name__ == "__main__":
    assess_labels("CNN_training\labels.csv")
    
    # asset = index_to_dir(1,0,1)
    
    # board = find_board(base_board, asset)[0]
    
    # points = [474.66241455078100,224.08909606933600,507.201416015625,209.0710906982420,547.2493896484380,303.3507080078130,514.7103881835940,316.7000427246090]
    
    # segment = BoardSegment("r", Point(*points[0:2]), Point(*points[2:4]), Point(*points[4:6]), Point(*points[6:8]), 1)
    
    # segment_image, id = segment.find_in_image(board, f"assets\coordinate_masks\image_{3}")
    
    # plt.imshow(segment_image)
    # plt.show()