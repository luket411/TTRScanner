import numpy as np
from os import path as ospath

dirs = [
    'Red', 'Blue', 'Green', 'Yellow', 'Black'
]
subdirs = np.array([
    'Yellow,Green,Gray', 'Blue,Black,Orange', 'White,Red,Pink'
])

colour_combos = {
    '2.0':['r'], #Red-Red
    '2.1':['y','g','tab:gray'], #Red-Yellow,Green,Gray
    '2.2':['b','k','tab:orange'], #Red-Blue,Black,Orange
    '2.3':['w','tab:pink'], #Red-White,Pink
    
    '3.0':['b'], #Blue-Blue
    '3.1':['y','g','tab:gray'], #Blue-Yellow,Green,Gray
    '3.2':['k','tab:orange'], #Blue-Black,Orange
    '3.3':['w','r','tab:pink'], #Blue-White,Red,Pink

    '4.0':['g'], #Green-Green
    '4.1':['y','tab:gray'], #Green-Yellow,Gray
    '4.2':['b','k','tab:orange'], #Green-Blue,Black,Orange
    '4.3':['w','r','tab:pink'], #Green-White,Red,Pink
    
    '5.0':['y'], #Yellow-Yellow
    '5.1':['g','tab:gray'], #Yellow-Green,Gray
    '5.2':['b','k','tab:orange'], #Yellow-Blue,Black,Orange
    '5.3':['w','r','tab:pink'], #Yellow-White,Red,Pink
    
    '6.0':['k'], #Black-Black
    '6.1':['y','g','tab:gray'], #Black-Yellow,Green,Gray
    '6.2':['b','tab:orange'], #Black-Blue,Orange
    '6.3':['w','r','tab:pink'], #Black-White,Red,Pink

}


def get_asset_dirs():
    full_dirs = []
    for num, dir_colour in enumerate(dirs, start=2):
        full_dirs.append(f'assets/{num}.0 {dir_colour}-{dir_colour}')
        for subnum, subdir in enumerate(subdirs, start=1):
            if dir_colour in subdir:
                iter_subdirs = subdir.split(',')
                iter_subdirs.remove(dir_colour)
                subdir = ",".join(iter_subdirs)
            full_dirs.append(f'assets/{num}.{subnum} {dir_colour}-{subdir}')
    return full_dirs

if __name__ == "__main__":
    print(get_asset_dirs())