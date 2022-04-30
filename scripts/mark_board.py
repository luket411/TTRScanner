from sys import path
from os import path as ospath


path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scripts.get_city_locations import show_city_clickable

from train_detection.Map import Map
from datasets.dataset import ImageFileDataset, index_to_dir
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


def assess(function, dataset = ImageFileDataset(1.0)):
    results = []
    for asset in dataset:
        print(asset)
        board = function(asset)
        show_city_clickable(board)
    return results

if __name__ == "__main__":
    assess(edge_detection)
    
    
    
