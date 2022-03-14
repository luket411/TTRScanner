from sys import path
from os import path as ospath
path.append(f'{ospath.dirname(__file__)}/..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as matplot_colours
from math import ceil
import cv2

from util.timer import timer
from util.geometry import Quadrilateral, Point
from util.constants import BASE_BACKGROUND_COLOUR

class BoardSegment(Quadrilateral):
    def __init__(self, base_colour, p1, p2, p3, p4, id, is_tunnel=False):
        super().__init__(p1, p2, p3, p4)
        self.id = id

        self.base_colour = base_colour
        self.is_tunnel = is_tunnel
        
        self.min_x_int = round(self.min_x)
        self.max_x_int = round(self.max_x)
        self.min_y_int = round(self.min_y)
        self.max_y_int = round(self.max_y)
        
        self.height = self.max_y - self.min_y
        self.width = self.max_x - self.min_x
        
        self.height_int = self.max_y_int - self.min_y_int
        self.width_int = self.max_x_int - self.min_x_int

    # Return num between 0 and 1 that the carriage has changed
    def containsCarriage(self, board):
        avg = self.getAvgColour(board)
        # print(f"Base_Colour: {self.base_colour}")
        # print(f"Average_Colour: {avg}")
        diff = self.base_colour - avg
        # print(f"diff: {diff}")
        return diff

    @timer
    def getAvgColour(self, image, show=False, snippet_output_file=None):
        avg_col = np.zeros((3), dtype=np.float32)
        area = 0
        
        # Create background image of light blue the size of the segment
        # out = np.full((self.height_int, self.width_int, 3), [209, 247, 255])
        out = np.empty((self.height_int, self.width_int, 3))

        # For this loop, 
        #   (i,j) are the coordinates for the scale of the segment
        #   (x,y) are the coordinates for the scale of the main image

        # Iterate through cols that contain the segment
        for i, y in enumerate(range(self.min_y_int, self.max_y_int)):

            # Create empty row
            row = np.full((self.width_int, 3), [209, 247, 255])

            # Iterate through rows that contain the segment
            for j, x in enumerate(range(self.min_x_int, self.max_x_int)):
                # If the current coordinate is within the segment, then add the value of that pixel to the new row
                if Point(x, y).within(self):
                    row[j] = image[y,x]
                    avg_col += image[y,x]
                    area += 1
            out[i] = row
    
        avg_col /= area

        if show or snippet_output_file is not None:
            _, plots = plt.subplots(3)

            plots[0].imshow(out.astype(np.float32))
            plots[1].imshow(self.get_filled_snippet(image, avg_col)[self.min_y_int:self.max_y_int, self.min_x_int:self.max_x_int])
            plots[2].imshow(self.get_filled_snippet(image, self.base_colour)[self.min_y_int:self.max_y_int, self.min_x_int:self.max_x_int])
            
            if show:
                plt.show()
            if snippet_output_file is not None:
                plt.savefig(snippet_output_file)

        return np.uint8(avg_col)

    def get_snippet_mask(self):
        out = np.full((self.height_int, self.width_int), 0, dtype=np.uint8)
        for i, y in enumerate(range(self.min_y_int, self.max_y_int)):
            row = np.full((self.width_int), 0, dtype=np.uint8)
            for j, x in enumerate(range(self.min_x_int, self.max_x_int)):
                if Point(x, y).within(self):
                    row[j] = 1
            out[i] = row
        return out

    def get_full_board_mask(self, board):
        full_board_mask = np.zeros(board.shape[0:2], dtype=np.uint8)
        full_board_mask[self.min_y_int:self.max_y_int, self.min_x_int:self.max_x_int] = self.get_snippet_mask()
        return full_board_mask

    def get_snippet_image(self, board):
        return cv2.bitwise_and(board, board, mask=self.get_full_board_mask(board))

    @timer
    def fill_segment(self, board, col):

        mask=self.get_full_board_mask(board)

        maskx3 = np.zeros((*mask.shape, 3))
        maskx3[:,:,0] = mask
        maskx3[:,:,1] = mask
        maskx3[:,:,2] = mask

        snipped_image = np.where(maskx3[:,:] == [0,0,0], board, col)

        return snipped_image

    def getPixels(self, image):
        np.zeros((self.max_x, self.max_y), dtype=np.float32)        
        return []
    
    def plot(self, image=None, show=False, label=False, fill_base=False, fill_avg=False):
        if image is not None:
            if fill_base:
                image = self.fill_segment(image, self.base_colour)
            elif fill_avg:
                image = self.fill_segment(image, self.getAvgColour(image))
            plt.imshow(image)
            plt.ylim(self.max_y_int, self.min_y_int)
            plt.xlim(self.max_x_int, self.min_x_int)
        if label:
            avg_x = np.average([self.min_x, self.max_x])
            avg_y = np.average([self.min_y, self.max_y])
            plt.text(avg_x, avg_y, self.id, fontsize='5')
        super().plot(colour=self.base_colour)
        if show:
            plt.show()

    def __repr__(self):
        return f"(BoardSegment:{self.id}, Base Colour:{self.base_colour})"