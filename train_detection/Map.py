from sys import path
from os import path as ospath

path.append(ospath.join(ospath.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from concurrent import futures

from train_detection.Connection import Connection
from train_detection.BoardSegment import BoardSegment
from train_detection.data_readers import read_base_colours_file, read_layout_csv, read_connection_breakdown
from datasets.dataset import index_to_dir
from board_handling.feature_detection import find_board
from util.constants import BASE_BACKGROUND_COLOUR
from util.timer import timer, timer_context

class Map():
    def __init__(self, segment_location='assets/0.0 Cropped/trains11.csv', connection_breakdown = 'assets/segment_info.csv', layout_colours='assets/0.0 Cropped/avg_colours11.csv'):

        self.connections: list(Connection) = []
        location_id = int(segment_location.split("trains")[1].split(".")[0])
        self.mask_location = f"assets\coordinate_masks\image_{location_id}"

        train_layouts = read_layout_csv(segment_location)
        base_colours = read_base_colours_file(layout_colours)

        for i, connection_info in enumerate(read_connection_breakdown(connection_breakdown)):
            city1 = connection_info[0][0]
            city2 = connection_info[0][1]
            connection_colour = train_layouts[connection_info[1][0]-1][0]
            segments = []
            for segment_index in connection_info[1]:
                coordinates = train_layouts[segment_index-1][1]
                # base_colour = base_colours[segment_index-1]
                base_colour = connection_colour
                segments.append(BoardSegment(base_colour, *coordinates, segment_index))
            
            self.connections.append(Connection(city1, city2, segments, connection_colour, i))

    def get_connection_by_id(self, id):
        for connection in self.connections:
            connection: Connection
            if connection.id == id:
                return connection
            
    def get_segments_made_from_connection(self, connection_id):
        connection: Connection = self.get_connection_by_id(connection_id)
        return [segment.id for segment in connection.segments]
    
    def get_segments(self):
        return [segment for connection in self.connections for segment in connection.segments]
    
    def get_segment_by_id(self, id):
        for segment in self.get_segments():
            if segment.id == id:
                return segment

    def plot(self, image=None, show=False, label=False):
        if image is not None:
            plt.imshow(image)

        connection: Connection
        for connection in self.connections:
            connection.plot(image=None, show=False, label=label)
        

        if show:
            plt.show()    

    def process_multicore(self, board, use_precompiled_masks=False):
        results = []
        
        mask_location = self.mask_location if use_precompiled_masks else None
        
        with futures.ProcessPoolExecutor() as executor:
            processes = [executor.submit(connection.hasTrain, board, mask_location) for connection in self.connections]
            for process in processes:
                results.append(process.result())
        return results

    def process_singlecore(self, board, use_precompiled_masks=False):
        results = []
        
        mask_location = self.mask_location if use_precompiled_masks else None
        
        for connection in self.connections:
            connection: Connection
            results.append(connection.hasTrain(board, mask_location=mask_location))
        return results

    def find_segments_in_image_full_calculations(self, board, debug_message=""):
        images = []
        with futures.ProcessPoolExecutor() as executor:
            processes = [executor.submit(segment.find_in_image, board) for segment in self.get_segments()]
            for process in processes:
                images.append(process.result())

        if debug_message:
            print(debug_message)

        return images

    def find_segments_in_image_quick(self, board, debug=False):
        segments = []
        for connection in self.connections:
            connection: Connection
            for segment in connection.segments:
                segment: BoardSegment
                segments.append(segment.find_in_image(board, self.mask_location))
                if debug:
                    print(f"Segment completed ({segment.id})")

        return segments


if __name__ == "__main__":
    board, _ = find_board("assets/0.0 Cropped/3.png", index_to_dir(1,0,1))
    empty_image = np.full((2000,3000, 3), BASE_BACKGROUND_COLOUR)
    map = Map()

    # map.process_multicore(board, use_precompiled_masks=False)

    map.process_multicore(board, use_precompiled_masks=True)

    # map.process_singlecore(board, use_precompiled_masks=False)

    # map.process_singlecore(board, use_precompiled_masks=True)