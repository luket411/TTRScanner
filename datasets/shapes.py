from unittest.mock import NonCallableMagicMock
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Quadrilateral():    
    def __init__(self, p1, p2, p3, p4):
        self.points = np.array([p1, p2, p3, p4], dtype=Point)
        
        self.max_x = np.array([p.x for p in self.points]).max(0)
        self.min_x = np.array([p.x for p in self.points]).min(0)

        self.max_y = np.array([p.y for p in self.points]).max(0)
        self.min_y = np.array([p.y for p in self.points]).min(0)

    
    def plot(self, image=None, show=True):
        if image is not None:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i in range(-1, len(self.points)-1):
            x1 = self.points[i].x
            y1 = self.points[i].y


            x2 = self.points[i+1].x
            y2 = self.points[i+1].y
            plt.plot((x1, x2), (y1, y2))
        if show:
            plt.show()
        
    def getEdgeLines(self):
        lines = []
        for i in range(-1, len(self.points)-1):
            lines.append(Line.construct_line_from_points(*self.points[i], *self.points[i+1]))
        return lines
        
    def isIn(self, point):
        if point.x > self.max_x or point.y > self.max_y or point.x < self.min_x or point.y < self.min_y:
            return False
        
        lines = self.getEdgeLines()
        vert_line = Line(np.inf,x_intercept=point.x)
        horz_line = Line(0, intercept=point.y)
        
        vert_line.plot(show=False)
        horz_line.plot(show=False)
        
        horz_bounds = []
        vert_bounds = []
        
        for line in lines:
            vert_intersection_point = vert_line.find_intersection(line)
            horz_intersection_point = horz_line.find_intersection(line)
            if vert_intersection_point is not None and vert_intersection_point.isValid() and self.min_y <= vert_intersection_point.y <= self.max_y:
                vert_bounds.append(vert_intersection_point)
            if horz_intersection_point is not None and horz_intersection_point.isValid() and self.min_x <= horz_intersection_point.x <= self.max_x:
                horz_bounds.append(horz_intersection_point)

        
        for p in [*vert_bounds, *horz_bounds]:
            p.plot(show=False)
        
        
        return isBetween(vert_bounds[0].y, vert_bounds[1].y, point.y) and isBetween(vert_bounds[0].x, vert_bounds[1].x, point.x)

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
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def plot(self, image=None, show=True):
        if image is not None:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        plt.scatter(self.x, self.y)
        
    def isValid(self):
        return self.x is not None and self.y is not None
        

class Line():
    def __init__(self, grad, intercept=None, x_intercept=None, min_x=-np.inf, min_y = -np.inf, max_x = np.inf, max_y = np.inf):
        self.grad = grad
        self.y_intercept = intercept
        self.x_intercept = x_intercept
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
            
    def get_y(self, x):
        if self.min_x < x < self.max_x:
            if self.is_vertical():
                return None
            elif self.is_horizontal():
                return self.y_intercept
            else:
                return self.grad*x + self.y_intercept
        
    def get_x(self, y):
        if self.min_y < y < self.max_y:
            if self.is_vertical():
                return self.x_intercept
            elif self.is_horizontal():
                return None
            else:
                return (y - self.y_intercept)/self.grad
    
    def __repr__(self):
        return f"(Grad: {self.grad}, y_intercept: {self.y_intercept}, x_intercept: {self.x_intercept})"

    def plot(self, image=None, show=True):
        
        line_points = []
        
        if self.is_horizontal():
            line_points = [(self.min_x, self.max_x), (self.get_y(self.min_x), self.get_y(self.max_x))]
        elif self.is_vertical():
            line_points = [(self.get_x(self.min_y), self.get_x(self.max_y)), (self.min_y, self.max_y)]
        else:
            line_points = [(self.get_x(self.min_y), self.get_x(self.min_y)), (self.get_y(self.min_x), self.get_y(self.max_x))]
        
        if image is not None:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        plt.plot(*line_points)
        
        if show:
            plt.show()
        

    def plot_cv(self, image, colour):
        (max_y, max_x, _) = image.shape
        min_x, min_y = 0, 0
        
        if self.is_horizontal():
            pt0 = (min_x, self.y_intercept)
            pt1 = (max_x, self.y_intercept)
        elif self.is_vertical():
            pt0 = (self.x_intercept, min_y)
            pt1 = (self.x_intercept, max_y)
        else:
            pt0 = (min_x, self.get_y(min_x))
            pt1 = (max_x, self.get_y(max_x))
            
        pt0 = (int(pt0[0]), int(pt0[1]))
        pt1 = (int(pt1[0]), int(pt1[1]))
            
        cv2.line(image, pt0, pt1, colour, 3)
        # print(f"Line Plotted: ({pt0}) -> ({pt1}) (Grad: {self.grad}, y_intercept: {self.y_intercept}, x_intercept: {self.x_intercept}, colour: {colour})")
        return image
    
    
    def is_vertical(self):
        return np.isinf(self.grad)
    
    
    def is_horizontal(self):
        return self.grad == 0
    
    
    def find_intersection(self, other_line):
    
        if self.is_vertical() and other_line.is_vertical(): # both are vertical
            return None
        if self.is_horizontal() and other_line.is_horizontal():
            return None
        
        if self.is_vertical():
            return Point(self.x_intercept, other_line.get_y(self.x_intercept))
        elif other_line.is_vertical():
            return Point(other_line.x_intercept, self.get_y(other_line.x_intercept))
        elif self.is_horizontal():
            return Point(other_line.get_x(self.y_intercept), self.y_intercept)
        elif other_line.is_horizontal():
            return Point(self.get_x(other_line.y_intercept), other_line.y_intercept)
        else:
            shared_x = (other_line.y_intercept - self.y_intercept)/(self.grad - other_line.grad)
            shared_y = self.get_y(shared_x)
            return Point(shared_x, shared_y)

    def construct_line_from_points(x0, y0, x1, y1):
        if x0 == x1:
            return Line(np.inf, x_intercept=x0)
        if y0 == y1:
            return Line(0, intercept=y0)
        grad = (y1-y0)/(x1-x0)
        intercept = y0 - grad*x0
        
        min_x = np.min([x0, x1])
        max_x = np.max([x0, x1])
        
        min_y = np.min([y0, y1])
        max_y = np.max([y0, y1])
        
        return Line(grad, intercept, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

# Returns true if z is between x and y
def isBetween(x,y,z):
    a = np.max([x,y])
    b = np.min([x,y])
    print(f"{a >= z >= b}:{a} >= {z} >= {b}")
    return a >= z >= b

if __name__ == "__main__":
    p4 = Point(1,4)
    p1 = Point(2,2)
    p2 = Point(4,2)
    p3 = Point(3,4)
    
    p5 = Point(1.75, 3)
    
    q = Quadrilateral(p1, p2, p3, p4)
    
    plt.xlim(q.min_x-1, q.max_x+1)
    plt.ylim(q.min_y-1, q.max_y+1)
    
    print(q.isIn(p5))
    q.plot(show=False)
    p5.plot(show=False)
    
    
    plt.show()