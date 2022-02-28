from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from csv import reader

from util.geometry import Point
from train_detection.BoardSegment import BoardSegment

base_img = "assets/0.0 Cropped/11.png"
train_layout = "assets/0.0 Cropped/trains11.csv"

COLOURS = [
    "r",                #0 - Red
    "g",                #1 - Green
    "tab:pink",         #2 - Pink
    "tab:orange",       #3 - Orange
    "k",                #4 - Black
    "tab:gray",         #5 - Grey
    "b",                #6 - Blue
    "y",                #7 - Yellow
    "w",                #8 - White
    "darkslategrey"     #9 - Uncategorised          
]

def read_layout_csv(layout_file):
    corner_data = []
    with open(layout_file) as open_file:
        file_reader = reader(open_file)
        for line in file_reader:
            if line[-1] == "":
                line[-1] = 9
            line = np.array(line, dtype=np.float32)
            p1 = Point(*line[1:3])
            p2 = Point(*line[3:5])
            p3 = Point(*line[5:7])
            p4 = Point(*line[7:9])
            colour = COLOURS[int(line[-1])]
            corner_data.append([colour, [p1, p2, p3, p4]])
    return corner_data

def plot_train_segments(train_segments):
    for segment in train_segments:
        segment.plot(show=False)

def get_train_segments(layout_file):
    corners = read_layout_csv(layout_file)
    train_segments = [BoardSegment(corner_set[0], *corner_set[1], i+1) for i, corner_set in enumerate(corners)]
    return train_segments

def label_image_train_segments(img=None, layout_file=train_layout):
    train_segments = get_train_segments(layout_file)
    
    if img is None:
        img = cv2.imread(base_img)
        img = cv2.cvtColor(img, 4)
    
    plt.imshow(img)
    plot_train_segments(train_segments)


if __name__ == "__main__":
    img = cv2.imread(base_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    train_segments = get_train_segments(train_layout)
    segment = train_segments[0]

    segment.getAvgColour(img)
    segment.plot(show=True, image=img)
    exit()
    label_image_train_segments()
    plt.show()
    
    
    img = np.full((2000,3000, 3), [209, 247, 255])
    label_image_train_segments(img)
    plt.show()
    