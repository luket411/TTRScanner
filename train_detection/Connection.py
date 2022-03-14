from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))

import numpy as np
import csv
import matplotlib.pyplot as plt

from train_detection.BoardSegment import BoardSegment

class Connection():
    def __init__(self, start, dest, segments, base_colour):
        self.start = start
        self.dest = dest
        self.segments: list(BoardSegment) = segments
        self.size = len(self.segments)
        self.base_colour = base_colour

    def __str__(self):
        return f"{self.start}->{self.dest}"
    
    def __repr__(self):
        return f"{self.start}->{self.dest}:{self.segments}"

    def plot(self, image=None, show=False, label=False, image_focus=None, fill_base=False, fill_avg=False, show_outline=True):
        for segment in self.segments:
            segment: BoardSegment

            if show_outline or label:
                segment.plot(label=label)

            if image is not None:
                if fill_base:
                    image = segment.fill_segment(image, segment.base_colour)
                elif fill_avg:
                    image = segment.fill_segment(image, segment.getAvgColour(image))

        
        if image is not None:

            if fill_base:
                image = segment.fill_segment(image, segment.base_colour)
            elif fill_avg:
                image = segment.fill_segment(image, segment.getAvgColour(image))

            plt.imshow(image)
            if image_focus:
                dimension_range = self.getDisplayRange()
                plt.xlim(dimension_range[0], dimension_range[1])
                plt.ylim(dimension_range[3], dimension_range[2])

        if show:
            plt.show()

    #ToDo: Implement function. Should return colour of pieces if there is one and False otherwise
    def hasTrain(self, board):
        return

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