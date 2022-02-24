import numpy as np
from util.geometry import Quadrilateral, Point


class BoardSegment(Quadrilateral):
    def __init__(self, colour, p1, p2, p3, p4, is_tunnel=False):
        super().__init__(p1, p2, p3, p4)
        self.colour = colour
        self.is_tunnel = is_tunnel

    def containsCarriage(self):
        return False

    def getColour(self):
        return self.colour

    def getLocation(self):
        return self.location

    def getPixels(self, image):
        return []
    
    def plot(self, image=None, show=False):
        super().plot(image=image, show=show, colour=self.colour)