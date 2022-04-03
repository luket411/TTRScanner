import numpy as np

BASE_BACKGROUND_COLOUR = np.array([209, 247, 255], dtype=np.uint8)
ENABLE_TIMER = True

COLOURS = {
    "r":"Red",                  #0 - Red
    "g":"Green",                #1 - Green
    "tab:pink":"Pink",          #2 - Pink
    "tab:orange":"Orange",      #3 - Orange
    "k":"Black",                #4 - Black
    "tab:gray":"Gray",          #5 - Grey
    "b":"Blue",                 #6 - Blue
    "y":"Yellow",               #7 - Yellow
    "w":"White"                 #8 - White
}

INVERSE_COLOURS = {
    "Red":[255,0,0],
    "Yellow":[255,255,0],
    "Blue":[0,0,255],
    "Orange":[255,135,0],
    "Green":[0,255,0],
    "White":[255,255,255],
    "Gray":[65,65,70],
    "Black":[0,0,0],
    "Pink":[255,50,240]
}