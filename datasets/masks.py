import csv
from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
   
    

def read_mask_from_location(segment_id, location):
    csv_file = ospath.join(location,f"{segment_id}.csv")
    return read_mask(csv_file)
    

def read_mask_from_fileset(segment_id, fileset):
    csv_file = f"assets/coordinate_masks/image_{fileset}/{segment_id}.csv"
    return read_mask(csv_file)
    
def read_mask(csv_file):
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        [min_y, min_x, max_y, max_x] = reader.__next__()
        cropped_mask = [line for line in reader]
        cropped_mask = np.array(cropped_mask, dtype=np.uint8)
    
    min_y, min_x, max_y, max_x = int(min_y), int(min_x), int(max_y), int(max_x)
    
    mask = np.zeros((2000,3000), dtype=np.uint8)
    mask[min_y:max_y, min_x:max_x] = cropped_mask
    
    return mask