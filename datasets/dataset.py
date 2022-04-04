from sys import path
from os import path as ospath, listdir
path.append(ospath.join(ospath.dirname(__file__), ".."))
import numpy as np
import cv2
from util.get_asset_dirs import dirs, subdirs
convertor = {
    1.0:"assets/1.0 Blank",
    1.1:"assets/1.1 Full Examples",
    2.0:"assets/2.0 Red-Red",
    2.1:"assets/2.1 Red-Yellow,Green,Gray",
    2.2:"assets/2.2 Red-Blue,Black,Orange",
    2.3:"assets/2.3 Red-White,Pink",
    3.0:"assets/3.0 Blue-Blue",
    3.1:"assets/3.1 Blue-Yellow,Green,Gray",
    3.2:"assets/3.2 Blue-Black,Orange",
    3.3:"assets/3.3 Blue-White,Red,Pink",
    4.0:"assets/4.0 Green-Green",
    4.1:"assets/4.1 Green-Yellow,Gray",
    4.2:"assets/4.2 Green-Blue,Black,Orange",
    4.3:"assets/4.3 Green-White,Red,Pink",
    5.0:"assets/5.0 Yellow-Yellow",
    5.1:"assets/5.1 Yellow-Green,Gray",
    5.2:"assets/5.2 Yellow-Blue,Black,Orange",
    5.3:"assets/5.3 Yellow-White,Red,Pink",
    6.0:"assets/6.0 Black-Black",
    6.1:"assets/6.1 Black-Yellow,Green,Gray",
    6.2:"assets/6.2 Black-Blue,Orange",
    6.3:"assets/6.3 Black-White,Red,Pink"
}

colours = {
    1:None,
    2:"Red",
    3:"Blue",
    4:"Green",
    5:"Yellow",
    6:"Black"
}

inv_cols = {col: idx for idx, col in colours.items()}

col_to_subnum = {
    "yellow":1,
    "green":1,
    "gray":1,
    "blue":2,
    "black":2,
    "orange":2,
    "white":3,
    "red":3,
    "pink":3

}


class ImageFileDataset():
    def __init__(self, dir_id, corners=None):

        self.colour = colours[int(str(dir_id)[0])]
       
        self.image_dir = convertor[dir_id]
        self.corners = corners

        self.images = [f"{self.image_dir}/{file}" for file in listdir(self.image_dir) if file[-4:] == '.jpg' or file[-4:] == '.png']
        
        self.index = -1
        self.size = len(self.images)
    
    def getAsset(self):
        return self.__next__()
    
    def getImageFileByKey(self, key):
        if f"{self.image_dir}/{key}.jpg" in self.images:
            return f"{self.image_dir}/{key}.jpg"
        elif f"{self.image_dir}/{key}.png" in self.images:
            return f"{self.image_dir}/{key}.png"
        else:
            raise KeyError(f"There is no item {key} in {self.image_dir}")
     
    def __iter__(self):
        return self
    
    def __next__(self):
         self.index += 1
         if self.index < self.size:
             return self.images[self.index]
         raise StopIteration

    def __getitem__(self, index):
        return self.images[index]

def index_to_dir(num, subnum, image_index):
    
    if num == 1:
        return f'assets/1.0 Blank/{image_index}.jpg'
    else:
        dir_colour = dirs[num-2]
        if subnum != 0:
            subdir_split = subdirs[subnum-1].split(",")
            if dir_colour in subdir_split:
                subdir_split.remove(dir_colour)
            subdir = ",".join(subdir_split)
        else:
            subdir = dir_colour
        return f'assets/{num}.{subnum} {dir_colour}-{subdir}/{image_index}.jpg'

def get_all_of_tile_colour(col: str):
    retVals = []
    col = col.lower()
    col_idx = col_to_subnum[col]
    for key in convertor.keys():
        if (key*10)%10 == col_idx:
            for asset in ImageFileDataset(key):
                retVals.append(asset)
    return retVals

def get_all_of_piece_colour(col: str):
    col = col.capitalize()
    col_idx = inv_cols[col]
    retVals = []
    for key in convertor.keys():
        if key//1 == col_idx:
            for asset in ImageFileDataset(key):
                retVals.append(asset)
    return retVals

if __name__ == "__main__":
    print(get_all_of_piece_colour("Black"))