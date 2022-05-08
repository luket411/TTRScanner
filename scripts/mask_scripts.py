from sys import path
from os import path as ospath

from datasets.masks import read_mask_from_location
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent import futures  

from train_detection.Map import Map
from train_detection.BoardSegment import BoardSegment

def create_and_write_mask(segment: BoardSegment, v):
    output_mask_dir = f"assets/coordinate_masks/image_{v}"
    mask = segment.get_mask_of_segment()
    
    cropped_mask = mask[segment.min_y_int:segment.max_y_int, segment.min_x_int:segment.max_x_int]
    
    
    csv_string = f"{segment.min_y_int},{segment.min_x_int},{segment.max_y_int},{segment.max_x_int}\n"
    for row in cropped_mask:
        csv_string += ",".join(row.astype(str))
        csv_string += "\n"
    
    csv_file = ospath.join(output_mask_dir, f"{segment.id}.csv")
    
    with open(csv_file, "w") as output_file:
        output_file.write(csv_string)

def generate_masks(v = 3):
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)

    with futures.ProcessPoolExecutor() as executor:
        processes = [executor.submit(create_and_write_mask, segment, v) for segment in map.get_segments()]
        for process in processes:
            process.result()

def test_system(v=3):
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    output_mask_dir = f"assets/coordinate_masks/image_{v}"
    
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)
    masks = []
    # with futures.ProcessPoolExecutor() as executor:
    #     processes = [executor.submit(segment.get_mask_of_segment) for segment in map.get_segments()]
    #     for process in processes:
    #         masks.append(process.result())
            
    for segment in map.get_segments():
        true_mask = segment.get_mask_of_segment()
        pre_compiled_mask = read_mask_from_location(segment.id, output_mask_dir)
        
        if not np.array_equal(true_mask, pre_compiled_mask):
            print(f"Masks do not equal for segment: {segment.id}")
            break
        else:
            print(f"Masks equal for segment: {segment.id}")