from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), "datasets"))
path.append(ospath.join(ospath.dirname(__file__), "board_handling"))
import cv2
import numpy as np
import matplotlib.pyplot as plt

counter = 0
coords = np.zeros((4,2), dtype=np.float32)
out_string = ""
train_counter = 212

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
    
    

avg_img_file = "assets/labelled.png"
avg_img = cv2.imread(avg_img_file, 1)
avg_img = cv2.cvtColor(avg_img, 4)
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.imshow(avg_img)
fig.canvas.mpl_connect('key_press_event', onclick)
try:
    plt.show()
    print(out_string)
except Exception:
    print(out_string)
    