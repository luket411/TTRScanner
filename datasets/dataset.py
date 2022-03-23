from sys import path
from os import path as ospath, listdir
path.append(ospath.join(ospath.dirname(__file__), ".."))
import numpy as np
import cv2

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

class ImageDataset():
    def __init__(self, image_dir):
        self.image_dir = image_dir
        if not ospath.exists(image_dir):
            raise Exception(f"image_dir: {image_dir} is empty")
        self.images_files = listdir(self.image_dir)
        
        self.images = np.empty((len(self.images_files)))
        
        for image_file in self.images_files:
            self.images(cv2.imread(image_file, ))
        
        self.index = 0
        self.size = len(self.images)
        
    def __iter__(self):
        return self
    
    def __next__(self):
         self.index += 1
         if self.index < self.size:
             return self.images[self.index]
         raise StopIteration

    def __getitem__(self, index):
        return self.images[index]


if __name__ == "__main__":
    images = ImageFileDataset("assets/1.0 Blank")
    for i in images:
        print(i)