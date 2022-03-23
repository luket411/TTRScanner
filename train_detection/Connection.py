from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))

import numpy as np
import csv
import matplotlib.pyplot as plt

from train_detection.BoardSegment import BoardSegment
from util.Counter import Counter
from util.constants import COLOURS

valid_train_colours = ["Black", "Red", "Yellow", "Green", "Blue"]

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

    def plot(self, image=None, show=False, label=False, image_focus=False, fill=False, use_avg_colour=False, use_colour=None):
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
            elif use_colour is not None:
                colour_val = use_colour

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

    def hasTrainResults(self, board):
        segment: BoardSegment
        
        connection_counter = Counter()
        for segment in self.segments:
            segment_counter = segment.containsCarriage(board)
            connection_counter.addVote(segment_counter.getWinner(), segment_counter.getWinningPercentage())
            # print(f"Carriage: {segment.id}, Colour: {self.base_colour}, Winner: {segment_counter.getWinner()}, ({round(segment_counter.getWinningPercentage()*100)}%)")
        
        # print(f"Connection: {str(self)}({self.id}), Base Colour: {self.base_colour}, Predicted Colour: {connection_counter.getWinner()}, Confidence: {round(connection_counter.getWinningPercentage(self.size)*100)}%")
        # print(f"Connection {str(self)} Completed")
        return [self.id, connection_counter]

    def hasTrain(self, board):
        result_counter = self.hasTrainResults(board)[1]
        predicted_colour = result_counter.getWinner()
        hasChanged = predicted_colour != COLOURS[self.base_colour]
        isTrainColour = predicted_colour in valid_train_colours        
        return [self.id, hasChanged and isTrainColour, predicted_colour]

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