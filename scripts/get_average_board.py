from sys import path
from os import path as ospath, listdir
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_and_save_corners import getImagesInDir
from board_handling.warp_board import annotate_fixed_city_points

def main():
    images = []
    for image_file in getImagesInDir("Assets/0.0 Cropped"):
        images.append(cv2.imread(image_file))
    
    avg = np.empty(images[0].shape, dtype=np.float32)
    for image in images:
        avg += (image/255)
    avg = avg/len(images)
    
    print(avg.shape)
    print(np.max(avg))
    print(np.min(avg))
    avg = cv2.cvtColor(avg, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"assets/avg.png", cv2.cvtColor(avg*255, cv2.COLOR_BGR2RGB))
    avg = annotate_fixed_city_points(avg)
    plt.imshow(avg)
    plt.show()
    
    
    
if __name__ == "__main__":
    main()