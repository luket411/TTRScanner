import numpy as np
from util.geometry import Quadrilateral, Point
import matplotlib.pyplot as plt
from math import ceil
import cv2
from os import listdir

class BoardSegment(Quadrilateral):
    def __init__(self, colour, p1, p2, p3, p4, is_tunnel=False):
        super().__init__(p1, p2, p3, p4)
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
        

    def containsCarriage(self):
        return False

    def getColour(self):
        return self.colour

    def getAvgColour(self, image, show=False):
        # plt.imshow(image[self.min_y_int:self.max_y_int, self.min_x_int:self.max_x_int])
        # plt.show()
        
        
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

        if show:
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
            # plt.show()
            file_name = len(listdir("assets\\general\\base"))
            plt.savefig(f"assets\\general\\base\\{file_name+1}.png")
        return avg_col


    def getPixels(self, image):
        np.zeros((self.max_x, self.max_y), dtype=np.float32)        
        return []
    
    def plot(self, image=None, show=False):
        super().plot(image=image, show=show, colour=self.colour)