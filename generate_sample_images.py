from os import listdir
import cv2
from train_detection.show_train_blocks import get_train_segments
import matplotlib.pyplot as plt
from util.timer import timer

@timer
def generate_sample_images(output_dir, image_file, train_data_file):
    
    train_segments = get_train_segments(train_data_file)
    
    img = cv2.imread(image_file, 1)
    img = cv2.cvtColor(img, 4)
    
    out_string_colours = ""
    
    for i, segment in enumerate(train_segments):
        try:
            
            file_name = len(listdir(output_dir))
            
            avg = segment.getAvgColour(img, snippet_output_file=f"{output_dir}\\{file_name+1}.png")
            
            print(f"{i+1}/{len(train_segments)}: {avg}")
            out_string_colours += f"{i+1},{avg[0]},{avg[1]},{avg[2]}\n"
        except KeyboardInterrupt:
            print (out_string_colours)
            exit()
    
    print(out_string_colours)


generate_sample_images(
    "assets\\general\\base",
    "assets\\0.0 Cropped\\11.png",
    "assets\\0.0 Cropped\\trains11.csv"
)