from sys import path
from os import getcwd, path as ospath
path.append(f'{ospath.dirname(__file__)}/..')
from csv import reader
import numpy as np
import matplotlib.pyplot as plt

from util.geometry import Point
from train_detection.BoardSegment import BoardSegment

## This file needs breaking up. Possibly into a Map (with Map and Connection classes), readers (with the three csv readers) and then leave the others in here as an example/ tester

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

class Connection():
    def __init__(self, start, dest, segments, base_colour):
        self.start = start
        self.dest = dest
        self.segments: list(BoardSegment) = segments
        self.size = len(self.segments)
        self.base_colour = base_colour

    def __str__(self):
        return f"{self.start}->{self.dest}:{self.segments}"
    
    def __repr__(self):
        return f"{self.start}->{self.dest}:{self.segments}"

    def plot(self, image=None, show=False, label=False):
        for segment in self.segments:
            segment.plot(label=label)
        
        if image is not None:
            plt.imshow(image)

        if show:
            plt.show()

    #ToDo: Implement function. Should return colour of pieces if there is one and False otherwise
    def hasTrain(self, board):
        return

class Map():
    def __init__(self, layout_info='assets/0.0 Cropped/trains11.csv', segment_info = 'assets/segment_info.csv', layout_colours='assets/0.0 Cropped/avg_colours11.csv'):

        self.connections: list(Connection) = []

        train_layouts = read_layout_csv(layout_info)
        base_colours = read_base_colours_file(layout_colours)

        for connection_info in read_segment_file(segment_info):
            city1 = connection_info[0][0]
            city2 = connection_info[0][1]
            segments = []
            for segment_index in connection_info[1]:
                colour = train_layouts[segment_index-1][0]
                coordinates = train_layouts[segment_index-1][1]
                base_colour = base_colours[segment_index-1]
                segments.append(BoardSegment(colour, *coordinates, segment_index))
            
            self.connections.append(Connection(city1, city2, segments, base_colour))

    def plot(self, image=None, show=False, label=False):
        for connection in self.connections:
            connection.plot(None, False, label)
        
        if image is not None:
            plt.imshow(image)

        if show:
            plt.show()

def validate_segment_dataset(filename):
    with open(filename) as open_file:
        file_reader = reader(open_file, quotechar='"')
        segments = []
        headers = file_reader.__next__()
        for line in file_reader:
            line = [segment for segment in line if segment]
            for segment in line[1:]:
                split_segments = segment.split(",")
                segments += split_segments
    
    segments = np.array(segments, dtype=int)

    segments = sorted(segments)
    if len(segments) != 300:
        print(f"Dataset not complete, {len(segments)}/300 found")
        return
    for i, segment in enumerate(segments):
        i += 1
        if i != segment:
            print(f"Dataset not valid, value {segment} is missing")
            return
    print("Dataset valid, all 300 segments accounted for")

def read_segment_file(segment_file):
    with open(segment_file) as open_file:
        file_reader = reader(open_file, quotechar='"')
        connections = []
        headers = file_reader.__next__()[1:]
        for line in file_reader:
            city1 = line[0]
            for i, segment in enumerate(line[1:]):
                if segment:
                    city2 = headers[i]
                    segment = np.array(segment.split(","), dtype=np.int16)
                    connections.append([[city1, city2], segment])
    return connections

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

def read_base_colours_file(colours_file):
    with open(colours_file) as open_file:
        colours = []
        file_reader = reader(open_file)
        for line in file_reader:
            colours.append(line[1:])
    
    return np.array(colours, dtype=np.float32)

if __name__ == "__main__":

    base_img = "assets/0.0 Cropped/11.png"
    empty_image = np.full((2000,3000, 3), [209, 247, 255])

    from board_handling.feature_detection import find_board
    train_layout = find_board(base_img, "assets/2.2 Red-Blue,Black,Orange/PXL_20220209_151954858.jpg")


    map = Map()
    map.plot(train_layout, True)