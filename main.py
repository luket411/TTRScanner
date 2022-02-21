from sys import path
from os import path as ospath, listdir
path.append(ospath.join(ospath.dirname(__file__), "datasets"))
path.append(ospath.join(ospath.dirname(__file__), "board_handling"))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from board_handling.transform_board import transform_board

counter = 0
coords = np.zeros((4,2), dtype=np.float32)

def onclick(event):
    global counter, coords
    if counter > 4:
        counter = 0
        print("\n\n\n")
        print("here")
    print(f"({event.xdata}, {event.ydata})")
    coords[counter] = np.array([event.xdata, event.ydata])
    counter += 1

def getFullPath(file_name):
    return ospath.join("Assets/1.0 Blank", file_name)

def getImagesInDir(directory):
    return [filename for filename in listdir(directory)]

def main():
    image_files = getImagesInDir("Assets/1.0 Blank")
    for image_file in image_files:
        image_file = getFullPath(image_file)
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.imshow(img)
        # fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onclick)
        plt.show()
        print(coords)
        warped_image = transform_board(img, coords)
        plt.imshow(warped_image)
        plt.show()
        break


if __name__ == "__main__":
    main()