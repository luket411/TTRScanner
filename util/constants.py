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
    "Green":[0,255,0],
    "Pink":[255,50,240],
    "Orange":[255,135,0],
    "Black":[0,0,0],
    "Gray":[65,65,70],
    "Blue":[0,0,255],
    "Yellow":[255,255,0],
    "White":[255,255,255]
}

INVERSE_CONNECTION_COLOURS = ["Gray","Gray","White","Yellow","Black","Black","Orange","Gray","Gray","Gray","Yellow","Gray","Blue","Gray","Yellow","Pink","Blue","Green","Red","Yellow","White","Orange","Gray","Gray","Orange","Black","Green","White","Green","Red","Black","Blue","Pink","Black","Pink","Red","Yellow","White","Orange","Green","Blue","Orange","Gray","Gray","Red","White","Red","Gray","Gray","Gray","Yellow","Gray","Blue","Green","Gray","Orange","Gray","Gray","Gray","Green","White","Gray","Gray","Gray","Pink","Red","Gray","Pink","Black","Gray","White","Pink","Blue","Green","Red","White","Black","Gray","Gray","Yellow","Pink","Orange","Blue","Gray","Gray","Green","Gray","Yellow","Gray","Gray","Gray","Gray","Gray","Orange","Gray","Red","Black","Gray","Gray","Pink","Blue"]
CONNECTION_COLOURS = ["tab:gray","tab:gray","w","y","k","k","tab:orange","tab:gray","tab:gray","tab:gray","y","tab:gray","b","tab:gray","y","tab:pink","b","g","r","y","w","tab:orange","tab:gray","tab:gray","tab:orange","k","g","w","g","r","k","b","tab:pink","k","tab:pink","r","y","w","tab:orange","g","b","tab:orange","tab:gray","tab:gray","r","w","r","tab:gray","tab:gray","tab:gray","y","tab:gray","b","g","tab:gray","tab:orange","tab:gray","tab:gray","tab:gray","g","w","tab:gray","tab:gray","tab:gray","tab:pink","r","tab:gray","tab:pink","k","tab:gray","w","tab:pink","b","g","r","w","k","tab:gray","tab:gray","y","tab:pink","tab:orange","b","tab:gray","tab:gray","g","tab:gray","y","tab:gray","tab:gray","tab:gray","tab:gray","tab:gray","tab:orange","tab:gray","r","k","tab:gray","tab:gray","tab:pink","b"]