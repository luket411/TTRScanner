from sys import path
from os import path as ospath
import matplotlib.pyplot as plt
from datasets.dataset import index_to_dir
from main import run_no_answers as TTRScanner

if __name__ == "__main__":
    
    # Use index_to_dir(x,y,z) to get the filename for a specific image in the assets/ folder.
    # e.g. index_to_dir(3,1,3) will return 'assets/3.1 Blue-Yellow,Green,Gray/3.jpg'
    asset = index_to_dir(1,1,1)
    
    # Will run TTRScanner on the asset specified. (Slightly Slow)
    # Shows a colour outline over a connection if it has identified a train is present
    TTRScanner(asset)