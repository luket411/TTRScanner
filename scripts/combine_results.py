from sys import path
from os import listdir, path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import numpy as np
import csv

col_index = {
        'Red':0,
        'Green':1,
        'Pink':2,
        'Orange':3,
        'Black':4,
        'Gray':5,
        'Blue':6,
        'Yellow':7,
        'White':8
}

def read_result_csv(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        reader.__next__()
        results = np.zeros((9,5))
        for result_row in reader:
            colour = col_index[result_row[0]]
            results[colour] = np.array([int(num) for num in result_row[1:]])

    return results

if __name__ == "__main__":
    results_path = "runs/21.04/Black Train Colour"
    files = [ospath.join(results_path, x) for x in listdir(results_path) if x[-3:] == "csv"]
    combined_results = np.zeros((9,5))
    for file in files:
        combined_results += read_result_csv(file)
    print(combined_results)