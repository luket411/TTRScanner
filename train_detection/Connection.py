from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))

import numpy as np
import csv
import matplotlib.pyplot as plt
import math

from train_detection.BoardSegment import BoardSegment

class Connection():
    def __init__(self, start, dest, segments, base_colour, id):
        self.start = start
        self.dest = dest
        self.segments = segments
        self.size = len(self.segments)
        self.base_colour = base_colour
        self.id = id

    def __str__(self):
        return f"{self.start}->{self.dest}"
    
    def __repr__(self):
        return f"({self.base_colour}):{self.start}->{self.dest}:{self.segments}"

    def plot(self, image=None, show=False, label=False, image_focus=None, fill=False, use_avg_colour=False):
        if image is not None:
            plt.imshow(image)    
            if image_focus:
                dimension_range = self.getDisplayRange()
                plt.xlim(dimension_range[0], dimension_range[1])
                plt.ylim(dimension_range[3], dimension_range[2])

        colour_val = None
        for segment in self.segments:
            segment: BoardSegment

            if use_avg_colour:
                colour_val = segment.getAverageColour(image) 

            params = dict(fill=fill, colour=colour_val)

            if label:
                params['label_connection'] = self.id

            # print(params)

            segment.plot(**params)
        
        if show:
            plt.show()

    def getSegmentBaseAvg(self):
        segment: BoardSegment
        outsum = []
        for segment in self.segments:
            outsum.append(segment.base_colour)
        return np.average(outsum, axis=0)

    def getPixels(self, board):
        pixels = []
        segment: BoardSegment
        for segment in self.segments:
            pixels.append(segment.getPixels(board))
        return pixels

    #ToDo: Implement function. Should return colour of pieces if there is one and False otherwise
    def hasTrain(self, board):
        segment: BoardSegment
        diff_sum = 0
        for segment in self.segments:
            diff_sum += abs(segment.containsCarriage(board, isGray=self.base_colour=="tab:gray"))
        diff_sum /= len(self.segments)
        return [self.id, diff_sum]

    def getDisplayRange(self):
        max_x, max_y = -np.inf, -np.inf
        min_x, min_y = np.inf, np.inf

        for segment in self.segments:
            if segment.min_x_int < min_x:
                min_x = segment.min_x_int

            if segment.min_y_int < min_y:
                min_y = segment.min_y_int
            
            if segment.max_x_int > max_x:
                max_x = segment.max_x_int
            
            if segment.max_y_int > max_y:
                max_y = segment.max_y_int
        
        return [min_x, max_x, min_y, max_y]