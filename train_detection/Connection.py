from sys import path
from os import path as ospath, listdir
path.append(ospath.join(ospath.dirname(__file__), ".."))

import matplotlib.pyplot as plt


class Connection():
    def __init__(self, start, dest, segments, base_colour):
        self.start = start
        self.dest = dest
        self.segments = segments
        self.size = len(self.segments)
        self.base_colour = base_colour

    def __str__(self):
        return f"{self.start}->{self.dest}:{self.segments}"
    
    def __repr__(self):
        return f"{self.start}->{self.dest}:{self.segments}"

    def plot(self, image=None, show=False, label=False):
        for segment in self.segments:
            segment.plot(label=label)
        
        if image is not None:
            plt.imshow(image)

        if show:
            plt.show()

    #ToDo: Implement function. Should return colour of pieces if there is one and False otherwise
    def hasTrain(self, board):
        return
