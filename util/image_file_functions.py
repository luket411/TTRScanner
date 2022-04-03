from os import path as ospath, listdir

def getFullPath(file_name, path="Assets/1.0 Blank"):
    return ospath.join(path, file_name)

def getImagesInDir(directory):
    return [getFullPath(filename, directory) for filename in listdir(directory) if filename[-4:] == ".png" or filename[-4:] == ".jpg"]