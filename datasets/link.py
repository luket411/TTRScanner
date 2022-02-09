import numpy as np
from shapes import Quadrilateral, Point


class BoardSegment():
    def __init__(self, colour, location, is_tunnel=False):
        self.colour = colour
        self.location = location

    def containsCarriage(self):
        return False

    def getColour(self):
        return self.colour

    def getLocation(self):
        return self.location

    def getPixels(self, image):
        return []