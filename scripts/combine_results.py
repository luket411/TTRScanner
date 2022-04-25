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

col_index_rev = [
        'Red',
        'Green',
        'Pink',
        'Orange',
        'Black',
        'Gray',
        'Blue',
        'Yellow',
        'White'
]

def read_result_csv(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        reader.__next__()
        results = np.zeros((9,5))
        for result_row in reader:
            colour = col_index[result_row[0]]
            results[colour] = np.array([int(num) for num in result_row[1:]])

    return results


def results_to_fp_fn(results):
    diff_results = np.zeros((9,4))
    diff_results[:,0:2] = results[:,0:2]
    diff_results[:,2] = results[:,2] + results[:,3]
    diff_results[:,3] = results[:,4]

    return diff_results

def results_to_fp_fn_percentage(results):
    fp_fn_res = results_to_fp_fn(results)
    for res_row in fp_fn_res:
        total = np.sum(res_row)
        res_row /= total
    return fp_fn_res

results_folders = ["runs/21.04/Train Colour X on tiles that arent also colour X", "runs/21.04/Train Colours X on tiles that are also colour X", "runs/21.04/Black Train Colour"]
results_path = results_folders[1]

def write_results(res_path, combined_results, header):
    report = header
    for col_idx, row in enumerate(combined_results):
        result_string = ",".join(list(row.astype(str)))
        report += f"{col_index_rev[col_idx]},{result_string}\n"

    out_file = ospath.join(results_path,res_path)
    
    with open(out_file, "w") as file:
        file.write(report)


if __name__ == "__main__":
    files = [ospath.join(results_path, x) for x in listdir(results_path) if x[-3:] == "csv" and x not in ["combined_results.csv", "fp_fn_results.csv"]]
    combined_results = np.zeros((9,5))
    for file in files:
        combined_results += read_result_csv(file)

    main_header = ",correctly_labelled,correctly_missed,labelled_wrong,shouldnt_be_labelled,missed\n"
    main_res_path = "combined_results.csv"

    fp_fn_header = ",positive,negative,false_positive,false_negative\n"
    fp_fn_path = "fp_fn_results.csv"
    fp_fn_results = results_to_fp_fn_percentage(combined_results)

    write_results(main_res_path, combined_results, main_header)
    write_results(fp_fn_path, fp_fn_results, fp_fn_header)
