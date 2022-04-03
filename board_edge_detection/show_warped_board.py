from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.cities import cities_loader

circle_colour = [0,0,255]


desired_corners = np.array([[0,0],
                [0,2000],
                [3000,2000],
                [3000,0]], dtype='float32')


transform_size = (3000, 2000)

def transform_board(board, original, desired, transform_size):
    M = cv2.getPerspectiveTransform(original, desired)
    transformed = cv2.warpPerspective(board, M, transform_size)

    # for p in desired_corners:
    #     cv2.circle(transformed, (int(p[0]), int(p[1])), 10, circle_colour, 20)

    
    return transformed

def annotate_fixed_city_points(transformed_board):
    cities = cities_loader()

    for _, location in cities:
        cv2.circle(transformed_board, (int(location[0]), int(location[1])), 60, circle_colour, 4)
    return transformed_board

def main():
    file = 'assets/IMG_4281.jpg'
    file2 = 'assets/IMG_1036.jpg'
    img = cv2.imread(file2, 1)

    corners_4821 = np.array([[830,700],[448,2174],[3582,2099],[3163,613]], dtype='float32')
    corners_1036 = np.array([[843.3,782.5],[243.5,2363.9],[3434,2814.7],[3333.1,960.5]], dtype='float32')
    corners_1939 = np.array([[613,800],[128,2794],[3969,2707],[3406,734]], dtype='float32')


    cities = cities_loader()
    city_locations_1036 = np.array([[1167,889],[414,2146],[3237.4,2510],[3196,1321]], dtype='float32')
    city_locations_1036_1 = np.array([[1386,1936],[1521,1693],[1738,1555],[1597,1446]], dtype='float32')
    city_locations_1036_2 = np.array([[414,2146], [622,2106], [2671, 1327], [2926, 1032]], dtype='float32')
    
    desired_city_locations = np.array([cities.edinburgh, cities.lisboa, cities.erzurum, cities.moskva], dtype='float32')
    desired_city_locations_1 = np.array([cities.marseille, cities.zurich, cities.munchen, cities.frankfurt], dtype='float32')
    desired_city_locations_2 = np.array([cities.lisboa, cities.madrid, cities.wilno, cities.petrograd], dtype='float32')
    
    # transformed = transform_board(img, city_locations_1036, desired_city_locations, transform_size)
    transformed = transform_board(img, corners_1036, desired_corners, transform_size)
    # transformed = annotate_fixed_city_points(transformed)
    return transformed

if __name__ == "__main__":
    plt.imshow(main())
    plt.show()