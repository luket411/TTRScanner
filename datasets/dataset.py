from os import path as ospath, listdir

class ImageDataset():
    def __init__(self, image_dir, corners=None):
        self.image_dir = image_dir
        self.corners = corners
        
        if not ospath.exists(image_dir):
            raise Exception(f"image_dir: {image_dir} is empty")
        self.images = listdir(self.image_dir)
        
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
    images = ImageDataset("assets\\1.0 Blank")
    for i in images:
        print(i)