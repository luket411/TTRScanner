from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.cities import cities_loader
import matplotlib.pyplot as plt

from util.geometry import Quadrilateral, Point

def get_colour_percentile(im, n):
    avg = [0,0,0]
    avg[0] = np.percentile(im[:,0].flatten(), n)
    avg[1] = np.percentile(im[:,1].flatten(), n)
    avg[2] = np.percentile(im[:,2].flatten(), n)
    return np.array(avg)

def main(img = None, lower_percentile=10):
    if img is None:
        img = cv2.imread('assets\clean_board.jpg', 1)
    ymax = img.shape[0]
    xmax = img.shape[1]
    n = 25
    selections = [
        [[xmax-n, 0], [xmax, ymax]],
        [[0,0], [n, ymax]],
        [[0,0], [xmax,n]],
        [[0,ymax-n], [xmax, ymax]]
    ]
    # selections = [
    #     [[2903, 127],[3025, 1546]]
    # ]


    boundary_boxes = []
    plt.imshow(cv2.cvtColor(img, 4))
    for [[a,b], [c,d]] in selections:
        p1 = Point(a,b)
        p2 = Point(c,b)
        p3 = Point(c,d)
        p4 = Point(a,d)
        quad = Quadrilateral(p1, p2, p3, p4)
        quad.plot()
        boundary_boxes.append(quad)
    plt.show()

    browns = np.empty((0,3), dtype=np.uint8)

    for [[bot_y, bot_x],[top_y, top_x]] in selections:
        colours_in_selection = img[bot_x:top_x,bot_y:top_y]
        shape = colours_in_selection.shape[:-1]
        dims = shape[0]*shape[1]
        colours_in_selection = colours_in_selection.reshape(dims, 3)
        browns = np.unique(np.concatenate((browns, colours_in_selection)), axis=0)

    browns_lower = get_colour_percentile(browns, lower_percentile)
    browns_upper = get_colour_percentile(browns, 100-lower_percentile)

    mask = cv2.inRange(img, browns_lower, browns_upper)
    mask = cv2.bitwise_not(mask)
    output = cv2.bitwise_and(img, img, mask=mask)
    return output


if __name__ == "__main__":
    figure = plt.figure(figsize=(10,8))
    img = cv2.imread('assets\clean_board.jpg', 1)
    for i in range(0,30,5):
        
        grey = main(img, i)
        
        figure.add_subplot(3, 2, (i//5)+1)
        plt.imshow(cv2.cvtColor(grey, cv2.COLOR_BGR2RGB))
        plt.title(i)
    plt.show()