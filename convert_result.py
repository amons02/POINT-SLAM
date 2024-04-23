import csv
import numpy as np

IN_FILE = 'MH01_GT_OG.csv'
OUT_FILE = 'MH01_GT_Converted.csv'
OUT_KEYS = ['#timestamp', ' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []',  ' q_RS_w []']
ZERO_NORM_COLS = [1, 2, 3, 4, 5, 6]

def read_data_to_dict(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        
        headers = next(csv_reader)
        
        data_dict = {header: [] for header in headers}
        
        for row in csv_reader:
            for header, value in zip(headers, row):
                data_dict[header].append(value.strip())
        
    return data_dict

def list_to_csv(data):
    f = open(OUT_FILE, "w")
    for i in range(len(data)):
        for j in range(len(data[i])):
            f.write('{:.5f}'.format(float(data[i][j])))
            if j < len(data[i]) - 1:
                f.write(" ")
        f.write("\n")
    f.close()


def zero_norm(data, column):
    data[column] = [float(x) for x in data[column]]
    col_first = data[column][0]

    for i in range(len(data[column])):
        data[column][i] -= col_first

    return data

def data_converter(data):
    out = []

    for key in OUT_KEYS:
        out.append(data[key])

    return out

def list_transpose(data):
    return np.array(data).T.tolist()

def append_ones_col(data):
    data.append([1 for i in range(len[data][0])])
    return data

def no_sci_not (data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = max(float(data[i][j]), 0.001)

    return data

# Example usage:
data = read_data_to_dict(IN_FILE)
OUT_FILE = 'MH01_GT_Converted.csv'


if __name__ == "__main__":
    data = read_data_to_dict(IN_FILE)
    data = data_converter(data)


    for col in ZERO_NORM_COLS:
        data = zero_norm(data, col)

    data = list_transpose(data)
    data = no_sci_not(data)

    list_to_csv(data)
