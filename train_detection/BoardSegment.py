from sys import path
from os import path as ospath
path.append(f'{ospath.dirname(__file__)}/..')

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import cv2

from util.geometry import Quadrilateral, Point
from util.timer import timer

class BoardSegment(Quadrilateral):
    def __init__(self, colour, p1, p2, p3, p4, id, is_tunnel=False):
        super().__init__(p1, p2, p3, p4)
        self.id = id
        self.colour = colour
        self.is_tunnel = is_tunnel
        
        self.min_x_int = round(self.min_x)
        self.max_x_int = round(self.max_x)
        self.min_y_int = round(self.min_y)
        self.max_y_int = round(self.max_y)
        
        self.height = self.max_y - self.min_y
        self.width = self.max_x - self.min_x
        
        self.height_int = ceil(self.height)
        self.width_int = ceil(self.width)
        self.avg_colour = None

    def containsCarriage(self):
        return False

    def getColour(self):
        return self.colour

    @timer
    def getAvgColour(self, image, show=False, snippet_output_file=None):      
        if self.avg_colour is None:
            avg_col = np.zeros((3), dtype=np.float32)
            area = 0
            
            out = np.full((self.height_int, self.width_int, 3), [209, 247, 255])
            for i, y in enumerate(range(self.min_y_int, self.max_y_int)):
                row = np.full((self.width_int, 3), [209, 247, 255])
                for j, x in enumerate(range(self.min_x_int, self.max_x_int)):
                    if self.isIn(Point(x,y)):
                        row[j] = image[y,x]
                        avg_col += image[y,x]
                        area += 1
                out[i] = row
        
            avg_col /= area
            self.avg_colour = avg_col

            if show or snippet_output_file is not None:
                _, plots = plt.subplots(2)
                plots[0].imshow(out)
                out = np.full((self.height_int, self.width_int, 3), [209, 247, 255])
                for i, y in enumerate(range(self.min_y_int, self.max_y_int)):
                    row = np.full((self.width_int, 3), [209, 247, 255])
                    for j, x in enumerate(range(self.min_x_int, self.max_x_int)):
                        if self.isIn(Point(x,y)):
                            row[j] = avg_col
                    out[i] = row

                plots[1].imshow(out)
                
                if show:
                    plt.show()
                if snippet_output_file is not None:
                    plt.savefig(snippet_output_file)

        return self.avg_colour


    def getPixels(self, image):
        np.zeros((self.max_x, self.max_y), dtype=np.float32)        
        return []
    
    def plot(self, image=None, show=False, label=False):
        if label:
            avg_x = np.average([self.min_x, self.max_x])
            avg_y = np.average([self.min_y, self.max_y])
            plt.text(avg_x, avg_y, self.id, fontsize='5')
        super().plot(image=image, show=show, colour=self.colour)

    def __repr__(self):
        return f"(BoardSegment:{self.id}, Colour:{self.colour})"