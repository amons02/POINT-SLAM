import csv
import numpy as np

IN_FILE = 'MH01_SUPERSLAM.csv'
OUT_FILE = 'MH01_SUPERscaled.csv'
ZERO_NORM_COLS = [1, 2, 3, 4, 5, 6]
SCALE = 2

import csv

def csv_rd(filename):
    data = list(csv.reader(open(filename), delimiter=' '))
    return list_transpose(data)

def list_to_csv(data):
    f = open(OUT_FILE, "w")
    for i in range(len(data)):
        for j in range(len(data[i])):
            f.write('{:.5f}'.format(float(data[i][j])))
            if j < len(data[i]) - 1:
                f.write(" ")
        f.write("\n")
    f.close()

def list_transpose(data):
    return np.array(data).T.tolist()

def scale_data(data, column):
    data[column] = [float(x) for x in data[column]]
    for i in range(len(data[column])):
        data[column][i] *= SCALE
    return data

if __name__ == "__main__":
    data = csv_rd(IN_FILE)

    for col in ZERO_NORM_COLS:
        data = scale_data(data, col)

    data = list_transpose(data)

    list_to_csv(data)

