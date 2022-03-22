from os import listdir, makedirs, path as ospath
import cv2
import numpy as np
import matplotlib.pyplot as plt

from board_handling.feature_detection import find_board

from train_detection.Connection import Connection
from train_detection.Map import Map
from train_detection.data_readers import read_connections, COLOURS

from util.timer import timer
from util.constants import BASE_BACKGROUND_COLOUR
from util.get_asset_dirs import get_asset_dirs

from datasets.dataset import ImageFileDataset

np.set_printoptions(suppress=True)

@timer
def main(target_file):
    v = 3
    base_file = f"assets/0.0 Cropped/{v}.png"
    train_location_data = f"assets/0.0 Cropped/trains{v}.csv"
    layout_colours = f"assets/0.0 Cropped/avg_colours{v}.csv"
    answers_file = "/".join(target_file.split("/")[:-1])
    target = int(target_file.split("/")[-1][0])
    
    board, base_board = find_board(base_file, target_file)
    map = Map(layout_colours=layout_colours, layout_info=train_location_data)
    blank_background = np.full(board.shape, BASE_BACKGROUND_COLOUR)
    answers = read_connections(answers_file)[target]

    connection: Connection
    target_file_split = target_file.split('/')
    main_num = target_file_split[1][:3]
    sub_num = target_file_split[2][0]


    # results = map.process_multicore(board)
    results=res
    plot_seperate_results(results, map, answers, target_file)



def plot_all_results(results, map, answers):
    sorted_results = sorted(results, key=lambda x:x[1])
    for pos, [idx, result] in enumerate(sorted_results):
        args = dict(
            c=map.connections[int(idx)].base_colour
        )
        
        if idx in answers:
            args['marker'] ='x'

        plt.scatter([pos], [result], **args)
    
    plt.show()


def plot_seperate_results(results, map, answers, title=""):
    sorted_results = sorted(results, key=lambda x:x[1])

    fig, subplots = plt.subplots(9, figsize=(24, 12))
    for pos, [idx, result] in enumerate(sorted_results):
        colour = map.connections[int(idx)].base_colour
        params = dict(
            c=colour
        )

        if idx in answers:
            params['marker'] = 'x'
        
        colour_index = COLOURS.index(colour)

        subplots[colour_index].scatter([pos], [result], **params)
        subplots[colour_index].annotate(str(idx), xy=(pos, result))
    
    for subplot in subplots:
        subplot.set_facecolor(BASE_BACKGROUND_COLOUR/255)
    if title:
        print(title)
        out_file = f'plots/{title}'
        out_dir = "/".join(out_file.split('/')[:-1])
        if not ospath.exists(out_dir):
            makedirs(out_dir)
        plt.savefig(out_file)
    # plt.show()


res = [[0, 2.697462320327759], [1, 4.058834075927734], [2, 38.01656977335612], [3, 3.5807367960611978], [4, 10.444445292154947], [5, 4.265151500701904], [6, 0.4064822196960449], [7, 6.126813888549805], [8, 1.9131151835123699], [9, 5.423974990844727], [10, 1.1289253234863281], [11, 5.750578880310059], [12, 6.306434631347656], [13, 0.9956741333007812], [14, 5.406494140625], [15, 74.81967568397522], [16, 0.9258613586425781], [17, 11.426631927490234], [18, 5.036272684733073], [19, 3.157385508219401], [20, 37.38317680358887], [21, 1.985795021057129], [22, 4.789915084838867], [23, 7.745098114013672], [24, 1.9676666259765625], [25, 60.0], [26, 5.658378601074219], [27, 36.424766540527344], [28, 9.17791748046875], [29, 3.2233705520629883], [30, 46.666666666666664], [31, 2.7631378173828125], [32, 70.0840368270874], [33, 17.873015721638996], [34, 14.968414306640625], [35, 2.960829973220825], [36, 2.666248321533203], [37, 36.93060302734375], [38, 1.9220517476399739], [39, 13.526880900065104], [40, 0.6524200439453125], [41, 3.8145338694254556], [42, 1.838150978088379], [43, 2.0361995697021484], [44, 62.77667808532715], [45, 38.21428298950195], [46, 12.52173900604248], [47, 4.155955632527669], [48, 4.822414875030518], [49, 4.069910049438477], [50, 2.0593929290771484], [51, 1.9238548278808594], [52, 42.94371795654297], [53, 33.9449577331543], [54, 4.242422103881836], [55, 1.2010478973388672], [56, 6.606060028076172], [57, 1.7758684158325195], [58, 7.411971092224121], [59, 21.894330978393555], [60, 44.30599880218506], [61, 3.630159378051758], [62, 5.975473403930664], [63, 3.86959171295166], [64, 66.28737258911133], [65, 62.395737965901695], [66, 2.350177764892578], [67, 8.397956848144531], [68, 50.625], [69, 4.56585693359375], [70, 8.428571701049805], [71, 14.791282653808594], [72, 0.6952133178710938], [73, 6.687602996826172], [74, 69.02165257930756], [75, 60.37454414367676], [76, 27.0], [77, 14.345238208770752], [78, 3.623377799987793], [79, 0.7223701477050781], [80, 17.219767252604168], [81, 1.5270449320475261], [82, 5.732276916503906], [83, 4.101730823516846], [84, 7.193473815917969], [85, 19.169679641723633], [86, 15.803987979888916], [87, 2.0951436360677085], [88, 3.296374797821045], [89, 5.90609073638916], [90, 0.7196969985961914], [91, 1.083333969116211], [92, 1.7499990463256836], [93, 1.933663050333659], [94, 2.6716904640197754], [95, 2.9583730697631836], [96, 7.854342142740886], [97, 2.8886547088623047], [98, 2.664335250854492], [99, 28.836354573567707], [100, 1.1521199544270833]]

@timer
def main2():
    for asset_dir in get_asset_dirs():
        dataset = ImageFileDataset(asset_dir)
        for asset in dataset:
            print(asset)
            main(asset)

if __name__ == "__main__":
    main2()