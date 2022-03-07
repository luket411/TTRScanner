from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
import matplotlib.pyplot as plt


class Line():
    def __init__(self, grad, intercept=np.nan, x_intercept=np.nan):
        self.grad = grad
        self.y_intercept = intercept
        if x_intercept is not np.nan:
            self.x_intercept = x_intercept
        else:
            self.x_intercept = -intercept/grad
        
    def get_y(self, x):
        return self.grad*x + self.y_intercept
    
    def get_x(self, y):
        return (y - self.y_intercept)/self.grad
    
    def __repr__(self):
        return f"(Grad: {self.grad}, y_intercept: {self.y_intercept}, x_intercept: {self.x_intercept})"

    def plot(self, image, colour):
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
            return (self.x_intercept, other_line.get_y(self.x_intercept))
        elif other_line.is_vertical():
            return (other_line.x_intercept, self.get_y(other_line.x_intercept))
        elif self.is_horizontal():
            return (other_line.get_x(self.y_intercept), self.y_intercept)
        elif other_line.is_horizontal():
            return (self.get_x(other_line.y_intercept), other_line.y_intercept)
        else:
            shared_x = (other_line.y_intercept - self.y_intercept)/(self.grad - other_line.grad)
            shared_y = self.get_y(shared_x)
            return (shared_x, shared_y)

    def construct_line_from_points(x0, y0, x1, y1):
        if x0 == x1:
            return Line(np.inf, np.nan, x0)
        grad = (y1-y0)/(x1-x0)
        intercept = y0 - grad*x0
        return Line(grad, intercept)



def plot_polar_line(rho, theta, show_plot=False):
    line = polar_to_cartesian(rho, theta)
    
    
    x0 = rho*cos(theta)
    x1 = x0-10
    x2 = x0+10
    
    y0 = line.get_y(x0)
    y1 = line.get_y(x1)
    y2 = line.get_y(x2)
    
    if show_plot:
        plt.plot((x1,y1), (x2,y2))
        plt.show()
        
    return (int(x0), int(y0)), (int(x1), int(y1)), (int(x2), int(y2))
    
def polar_to_cartesian(rho, theta):
    grad =  -cos(theta)/sin(theta)
    intercept = rho/sin(theta)
    line = Line(grad, intercept)
    return line


if __name__ == "__main__":
    p0, _, _ = plot_polar_line(319, 1.652)
    line = polar_to_cartesian(319, 1.652)
    img_file = "assets/IMG_4281.jpg"
    img = cv2.imread(img_file, 1)
    
    black_screen = np.zeros(img.shape, dtype=np.uint8)
    
    p1 = (0,int(line.get_y(0)))
    p2 = (3000, int(line.get_y(3000)))
    
    
    cv2.line(img, p1, p2, (0,0,255), 10)
    
    plt.imshow(img)
    plt.show()
    