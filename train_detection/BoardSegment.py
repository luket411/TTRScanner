from sys import path
from os import path as ospath
path.append(f'{ospath.dirname(__file__)}/..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as matplot_colours
from math import ceil
import cv2

from util.timer import timer
from util.geometry import Quadrilateral, Point, isBetween
from util.constants import BASE_BACKGROUND_COLOUR, COLOURS, INVERSE_COLOURS
from util.Counter import Counter

colour_ranges = {
    "Black",
    "White",
    "Yellow",
    "Orange",
    "Green",
    "Blue",
    "Gray",
    "Pink",
    "Red"
}

class BoardSegment(Quadrilateral):
    def __init__(self, base_colour, p1, p2, p3, p4, id, is_tunnel=False):
        super().__init__(p1, p2, p3, p4)
        self.id = id

        self.base_colour = COLOURS[base_colour] # matlab format
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
    def containsCarriage(self, board, show_breakdown=False):
        pixels_in_carriage = self.getPixels(board)
        pixels_in_carriage = np.array([pixels_in_carriage], dtype=np.uint8)
        hsv_pixels_in_carriage = cv2.cvtColor(pixels_in_carriage, cv2.COLOR_RGB2HSV).squeeze(0)
        
        hues = hsv_pixels_in_carriage[:,0]
        sats = hsv_pixels_in_carriage[:,1]
        vals = hsv_pixels_in_carriage[:2]
        
        # print(f"hues, min:{hues.min()}, max:{hues.max()}, median:{np.median(hues)}, average:{np.average(hues)}")
        # print(f"sats, min:{sats.min()}, max:{sats.max()}, median:{np.median(sats)}, average:{np.average(sats)}")
        # print(f"vals, min:{vals.min()}, max:{vals.max()}, median:{np.median(vals)}, average:{np.average(vals)}")
        
        pixel_display = np.full(hsv_pixels_in_carriage.shape, BASE_BACKGROUND_COLOUR)
        counter = Counter()
        
        for pixel_idx, [h, s, v] in enumerate(hsv_pixels_in_carriage):
                
            if isBetween(0, 60, s) and self.base_colour == "White":
                counter.addVote("White")
                pixel_display[pixel_idx] = [255,255,255]
                
            elif isBetween(0, 60, s) and self.base_colour == "Gray":
                counter.addVote("Gray")
                pixel_display[pixel_idx] = [65,65,70]
                
            elif isBetween(8, 20, h) and self.base_colour == "Orange":
                counter.addVote("Orange")
                pixel_display[pixel_idx] = [255,135,0]
                
            elif isBetween(130, 175, h) and self.base_colour == "Pink":
                counter.addVote("Pink")
                pixel_display[pixel_idx] = [255,50,240]
                
            elif isBetween(30, 90, h):
                counter.addVote("Green")
                pixel_display[pixel_idx] = [0,255,0]

            elif v < 50:
                counter.addVote("Black")
                pixel_display[pixel_idx] = [0,0,0]
                
            elif isBetween(20, 30, h):
                counter.addVote("Yellow")
                pixel_display[pixel_idx] = [255,255,0]
                
            elif isBetween(100, 130, h):
                counter.addVote("Blue")
                pixel_display[pixel_idx] = [0,0,255]
            
            elif h < 8 or h > 175:
                counter.addVote("Red")
                pixel_display[pixel_idx] = [255,0,0]
        
        # print(f"Carriage: {self.id}, Winner: {counter.getWinner()}, ({round(counter.getWinningPercentage(len(hsv_pixels_in_carriage))*100)}%)")
        # print(f"Votes Cast: {counter.getTotalVotes()}")
        # counter.printBreakdown(len(hsv_pixels_in_carriage))
        
        
        if show_breakdown:        
            pixel_display_tall = np.full((200, *hsv_pixels_in_carriage.shape), pixel_display)
            pixel_value_tall = np.full((200, *hsv_pixels_in_carriage.shape), pixels_in_carriage)
            display = np.vstack((pixel_display_tall, pixel_value_tall))
            plt.imshow(display)
            plt.show()
        
        return counter

    # Deprecated
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

        # if show or snippet_output_file is not None:
        #     _, plots = plt.subplots(3)

        #     plots[0].imshow(out.astype(np.float32))
        #     plots[1].imshow(self.fill_segment(image, avg_col)[self.min_y_int:self.max_y_int, self.min_x_int:self.max_x_int])
        #     plots[2].imshow(self.fill_segment(image, self.base_colour)[self.min_y_int:self.max_y_int, self.min_x_int:self.max_x_int])
            
        #     if show:
        #         plt.show()
        #     if snippet_output_file is not None:
        #         plt.savefig(snippet_output_file)

        return np.uint8(avg_col)


    def getPixels(self, image):
        pixels = []
        for i, y in enumerate(range(self.min_y_int, self.max_y_int)):
            for j, x in enumerate(range(self.min_x_int, self.max_x_int)):
                if Point(x, y).within(self):
                    pixels.append(image[y,x])
        return np.uint8(pixels)
    
    def getAverageColour(self, image):
        return np.average(self.getPixels(image), axis=0)

    def getMedianColour(self, image):
        return np.median(self.getPixels(image), axis=0)
    
    def plot(self, image=None, show=False, label=False, fill=False, colour=None, label_connection=None):

        colour_val = np.array(INVERSE_COLOURS[self.base_colour])
        if colour is not None:
            colour_val = colour
        
        if image is not None:
            plt.imshow(image)
            plt.ylim(self.max_y_int, self.min_y_int)
            plt.xlim(self.max_x_int, self.min_x_int)
        
                
            
        if label or label_connection:
            params = {'fontsize':10}
            params['x'] = np.average([self.min_x, self.max_x])
            params['y'] = np.average([self.min_y, self.max_y])
            if label_connection:
                params['s'] = label_connection
            else:
                params['s'] = self.id
            plt.text(**params)
        
        super().plot(fill=fill, colour=colour_val)

        if show:
            plt.show()

    def __repr__(self):
        return f"(BoardSegment:{self.id}, Base Colour:{self.base_colour})"