import math
from sys import path
from os import path as ospath

path.append(ospath.join(ospath.dirname(__file__), ".."))
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from util.geometry import Line
from board_edge_detection.full_board_pipeline import main as edge_detection_raw
from board_handling.warp_board import annotate_fixed_city_points
from board_handling.feature_detection import find_board as feature_detection_raw

def feature_detection(asset):
    base_asset = "assets/0.0 Cropped/11.png"
    board = feature_detection_raw(base_asset, asset)[0]
    return annotate_fixed_city_points(board, "assets/0.0 Cropped/cities11.csv")

def edge_detection(asset):
    board = edge_detection_raw(asset)
    return cv2.cvtColor(board, 4)

def read_city_csv(input_csv):
    with open(input_csv, "r") as file:
        reader = csv.reader(file)
        results = [line for line in reader]
    return results

def euclid_distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)
    

def main(input_csv, function, file):
    results = read_city_csv(input_csv)
    base_results = read_city_csv("assets/0.0 Cropped/cities11.csv")
    board = function(file)
    annotated_board = annotate_fixed_city_points(board, "assets/0.0 Cropped/cities11.csv")
    plt.imshow(annotated_board)
    total_distance = 0
    for city_index in range(len(results)):
        results_coords = np.array(results[city_index][1:], dtype=np.float32)
        base_coords = np.array(base_results[city_index][1:], dtype=np.float32)
        distance = euclid_distance(*results_coords, *base_coords)
        print(f"Distance for {results[city_index][0]}: {distance}")
        total_distance += distance
        # plt.plot((results_coords[0], base_coords[0]), (results_coords[1], base_coords[1]), color=[0,1,0])
    print(f"Avg distance: {total_distance/len(results)}")
    plt.show()
    


if __name__ == "__main__":
    features = ("runs/feature_Detection/5.csv", feature_detection)
    edge = ("runs/edge_Detection/5.csv", edge_detection)
    main(*features, "assets/1.0 Blank/5.jpg")