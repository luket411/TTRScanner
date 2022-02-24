from sys import path
from os import path as ospath, listdir
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from csv import reader

from util.geometry import Point
from train_detection.BoardSegment import BoardSegment

# base_img = "assets\\0.0 Cropped\\11.png"
base_img = "assets\\labelled.png"
train_layout = "assets\\0.0 Cropped\\trains11.csv"

COLOURS = [
    "r", #0 - Red
    "g", #1 - Green
    "m", #2 - Pink
    "c", #3 - Orange
    "k", #4 - Black
    "w", #5 - Grey
    "b", #6 - Blue
    "y"  #7 - Yellow
]

def read_layout_csv(layout_file):
    corner_data = []
    with open(layout_file) as open_file:
        file_reader = reader(open_file)
        for line in file_reader:
            line = np.array(line, dtype=np.float32)
            p1 = Point(*line[1:3])
            p2 = Point(*line[3:5])
            p3 = Point(*line[5:7])
            p4 = Point(*line[7:9])
            colour = COLOURS[int(line[-1])]
            corner_data.append([colour, [p1, p2, p3, p4]])
    return corner_data

def label_train_segments(train_segments):
    for segment in train_segments:
        segment.plot(show=False)


def main(img=None, img_file=base_img, layout_file=train_layout):
    corners = read_layout_csv(layout_file)
    if img is None:
        img = cv2.imread(base_img)
        img = cv2.cvtColor(img, 4)
    train_segments = [BoardSegment(corner_set[0], *corner_set[1]) for corner_set in corners]
    plt.imshow(img)
    label_train_segments(train_segments)
    

if __name__ == "__main__":
    main()
    plt.show()
    