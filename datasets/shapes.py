from unittest.mock import NonCallableMagicMock
import numpy as np
import matplotlib.pyplot as plt

class Quadrilateral():    
    def __init__(self, p1, p2, p3, p4):
        self.points = np.array([p1, p2, p3, p4], dtype=Point)
        
        self.max_x = np.array([p.x for p in self.points]).max(0)
        self.min_x = np.array([p.x for p in self.points]).min(0)

        self.max_y = np.array([p.y for p in self.points]).max(0)
        self.min_y = np.array([p.y for p in self.points]).min(0)

    
    def show(self, image=None):
        if image != None:
            plt.imshow(image)
        for i in range(-1, len(self.points)-1):
            x1 = self.points[i].x
            x2 = self.points[i+1].x

            y1 = self.points[i].y
            y2 = self.points[i+1].y
            plt.plot((x1, x2), (y1, y2))
        
        plt.show()




class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getPoints(self):
        return (self.x, self.y)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    p4 = Point(1,4)
    p1 = Point(2,2)
    p2 = Point(4,2)
    p3 = Point(3,4)

    q = Quadrilateral(p1, p2, p3, p4)
    q.show()