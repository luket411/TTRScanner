from sys import path
from os import path as ospath

path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt

from train_detection.Map import Map
counter = 0
coords = np.zeros((4,2), dtype=np.float32)
out_string = ""
train_counter = 93

def onclick(event):
    global counter, coords, out_string, train_counter
        
    print(f"({event.xdata}, {event.ydata})")
    coords[counter] = np.array([event.xdata, event.ydata])
    counter += 1

    if counter >= 4:
        counter = 0
        train_counter += 1
        print(selection := f"{train_counter}, {coords[0,0]}, {coords[0,1]}, {coords[1,0]}, {coords[1,1]}, {coords[2,0]}, {coords[2,1]}, {coords[3,0]}, {coords[3,1]},\n")
        out_string += selection
    
    

img_file = "assets/0.0 Cropped/3.png"

img = cv2.imread(img_file, 1)
img = cv2.cvtColor(img, 4)


fig = plt.figure(1)




ax = fig.add_subplot(111)
# ax.imshow(img)
map = Map(layout_info="assets/0.0 Cropped/trains3.csv")
map.plot(image=img)
fig.canvas.mpl_connect('key_press_event', onclick)
try:
    plt.show()
    print(out_string)
except Exception:
    print(out_string)
    