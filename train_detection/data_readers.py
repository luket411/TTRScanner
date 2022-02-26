from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))

import numpy as np
from csv import reader

from util.geometry import Point

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


if __name__ == "__main__":
    validate_segment_dataset('assets/segment_info.csv')