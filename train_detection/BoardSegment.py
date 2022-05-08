from sys import path
from os import path as ospath


path.append(f'{ospath.dirname(__file__)}/..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as matplot_colours
from math import ceil
import cv2

from train_detection.data_readers import read_layout_csv
from board_handling.feature_detection import find_board
from datasets.dataset import index_to_dir
from datasets.masks import read_mask_from_location
from util.timer import timer, timer_context
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

    def contains_carriage_full_calculations(self, board, show_breakdown=False):
        pixels_in_carriage = self.getPixels(board)
        return self.process_pixels(show_breakdown, pixels_in_carriage)

    def contains_carriage_quick(self, board, mask_location, show_breakdown=False):
        pixels_in_carriage = self.getPixelsFromMask(board, mask_location)
        return self.process_pixels(show_breakdown, pixels_in_carriage)

    def process_pixels(self, show_breakdown, pixels_in_carriage):
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


    def getPixels(self, image):
        pixels = []
        for i, y in enumerate(range(self.min_y_int, self.max_y_int)):
            for j, x in enumerate(range(self.min_x_int, self.max_x_int)):
                if Point(x, y).within(self):
                    pixels.append(image[y,x])
        return np.uint8(pixels)
    
    def getPixelsFromMask(self, image, mask_location):
        mask = read_mask_from_location(self.id, mask_location)
        
        masked = cv2.bitwise_and(image, image, mask=mask)
        masked_2 = []

        for i, y in enumerate(range(self.min_y_int, self.max_y_int)):
            for j, x in enumerate(range(self.min_x_int, self.max_x_int)):
                if not np.array_equal([0,0,0], masked[y,x]):
                    masked_2.append(masked[y,x])

        # masked = masked[np.any(masked != [0, 0, 0], axis=-1)]

        return masked_2
        
    
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

    def get_mask_of_segment(self, board_shape=(2000,3000)):
        mask = np.zeros(board_shape, dtype="uint8")
        for x in range(self.min_x_int, self.max_x_int):
            for y in range(self.min_y_int, self.max_y_int):
                if Point(x, y).within(self):
                    mask[y, x] = 255
                    
        return mask

    def find_in_image(self, board, mask_location=None):

        if mask_location is None:
            mask = np.zeros(board.shape[:2], dtype="uint8") 
            for x in range(self.min_x_int, self.max_x_int):
                for y in range(self.min_y_int, self.max_y_int):
                    if Point(x, y).within(self):
                        mask[y, x] = 255
                        

        else:
            mask = read_mask_from_location(self.id, mask_location)

        masked = cv2.bitwise_and(board, board, mask=mask)
        
        image_radius = 75
        (y_size, x_size) = board.shape[0:2]
        
        avg_x = (self.max_x_int + self.min_x_int)//2
        avg_y = (self.max_y_int + self.min_y_int)//2
        
        if avg_x < image_radius:
            min_x, max_x = 0, 2*image_radius
        elif avg_x > x_size - image_radius:
            min_x, max_x = x_size - 2*image_radius, x_size
        else:
            min_x, max_x = avg_x - image_radius, avg_x + image_radius
        
        if avg_y < image_radius:
            min_y, max_y = 0, 2*image_radius
        elif avg_y > y_size - image_radius:
            min_y, max_y = y_size - 2*image_radius, y_size
        else:
            min_y, max_y = avg_y - image_radius, avg_y + image_radius
        
        return masked[min_y:max_y, min_x:max_x], self.id

    def __repr__(self):
        return f"(BoardSegment:{self.id}, Base Colour:{self.base_colour})"
    
if __name__ == "__main__":
    board, _ = find_board("assets/0.0 Cropped/3.png", index_to_dir(1,0,1))
    mask_location = f"assets\coordinate_masks\image_3"
    
    segment_id = 1
    
    connections = read_layout_csv("assets/0.0 Cropped/trains3.csv")
    points = connections[segment_id-1][1]
    
    segment = BoardSegment("r", *points, 1)    

    print("==============")

    with timer_context("Full Calc getPixels"):
        segment.getPixels(board)
    
    with timer_context("Quick Calc getPixels"):
        segment.getPixelsFromMask(board, mask_location)
    
    print("==============")

    with timer_context("Full Calc contains carriage"):
        segment.contains_carriage_full_calculations(board)
    
    with timer_context("Quick Calc contains carriage"):
        segment.contains_carriage_quick(board, mask_location)
    
    print("==============")

    with timer_context("Full Calc find snippet"):
        segment.find_in_image(board)

    with timer_context("Quick Calc find snippet"):
        segment.find_in_image(board, mask_location)
    