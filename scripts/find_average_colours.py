from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt

from train_detection.data_readers import read_layout_csv
from train_detection.BoardSegment import BoardSegment
from util.geometry import Point

def main(source_image, layout_file='assets/0.0 Cropped/trains11.csv'):
    
    outstring =""
    
    layout = read_layout_csv(layout_file)
    for i, corner_set in enumerate(layout):
        base_colour = corner_set[0]
        corner_set = corner_set[1]
        points = [Point(x,y) for (x,y) in corner_set]
        b = BoardSegment(base_colour, *points, i+1)
        avg_col = b.getAverageColour(source_image)
        print(iter:= f"{i+1},{avg_col[0]},{avg_col[1]},{avg_col[2]}\n")
        outstring += iter
        
        b.plot(image=source_image, colour=avg_col)
        
        # break
    
    print(outstring)
    

if __name__ == "__main__":
    img = cv2.imread('assets/0.0 Cropped/3.png', 1)
    img = cv2.cvtColor(img, 4)
    main(img)