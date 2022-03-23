from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from concurrent import futures

from train_detection.Connection import Connection
from train_detection.BoardSegment import BoardSegment
from train_detection.data_readers import read_base_colours_file, read_layout_csv, read_segment_file
from util.timer import timer

class Map():
    def __init__(self, layout_info='assets/0.0 Cropped/trains11.csv', segment_info = 'assets/segment_info.csv', layout_colours='assets/0.0 Cropped/avg_colours11.csv'):

        self.connections: list(Connection) = []

        train_layouts = read_layout_csv(layout_info)
        base_colours = read_base_colours_file(layout_colours)

        for i, connection_info in enumerate(read_segment_file(segment_info)):
            city1 = connection_info[0][0]
            city2 = connection_info[0][1]
            connection_colour = train_layouts[connection_info[1][0]-1][0]
            segments = []
            for segment_index in connection_info[1]:
                coordinates = train_layouts[segment_index-1][1]
                base_colour = base_colours[segment_index-1]
                segments.append(BoardSegment(base_colour, *coordinates, segment_index))
            
            self.connections.append(Connection(city1, city2, segments, connection_colour, i))

    def plot(self, image=None, show=False, label=False):
        if image is not None:
            plt.imshow(image)

        connection: Connection
        for connection in self.connections:
            connection.plot(image=None, show=False, label=label)
        

        if show:
            plt.show()    
    
    @timer
    def process_multicore_results(self, board):
        results = []
        with futures.ProcessPoolExecutor() as executor:
            processes = [executor.submit(connection.hasTrainResults, board) for connection in self.connections]
            for process in processes:
                results.append(process.result())
        return results

    def process_multicore(self, board):
        results = []
        with futures.ProcessPoolExecutor() as executor:
            processes = [executor.submit(connection.hasTrain, board) for connection in self.connections]
            for process in processes:
                results.append(process.result())
        return results

    @timer
    def process_singlecore(self, board):
        results = []
        for connection in self.connections:
            results.append(connection.hasTrain(board))
        return results
    
if __name__ == "__main__":
    
    base_img = "assets/0.0 Cropped/11.png"
    empty_image = np.full((2000,3000, 3), [209, 247, 255])

    from board_handling.feature_detection import find_board
    train_layout = find_board(base_img, "assets/2.2 Red-Blue,Black,Orange/PXL_20220209_151954858.jpg")


    map = Map()
    map.plot(train_layout, True)