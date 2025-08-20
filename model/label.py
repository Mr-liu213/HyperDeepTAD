import numpy as np
import os


def load_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [int(line.strip()) for line in file]


def load_all_lines_as_matrix(file2):
    lines = []
    with open(file2, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append([int(value) for value in line.strip().split()])
    return np.array(lines, dtype=np.int32)  

# Sort by column maximum value
def sort_matrix_by_column_max(matrix):
    max_values = np.max(matrix, axis=0)
    sorted_indices = np.argsort(-max_values)
    return matrix[:, sorted_indices]

# Sliding window row by row, extract and perform local sorting, then save to a file
def split_matrix_by_sliding_window(large_matrix, sub_matrix_size, labels, save_dir, chr_name, res):
    n, m = large_matrix.shape
    p, q = sub_matrix_size  

    for i in range(n - p + 1):  
        sub_matrix = large_matrix[i:i+p, :]
        if np.all(sub_matrix == 0):
            continue
      
        sorted_sub_matrix = sort_matrix_by_column_max(sub_matrix)
        sub_matrix = sorted_sub_matrix[:, :q]

        
        middle_row_index = i + p // 2 + 1  

        #label
        label_flag = 1 if middle_row_index in labels else 0

        save_sub_matrix_to_txt(sub_matrix, middle_row_index, label_flag, save_dir, chr_name, res)


def save_sub_matrix_to_txt(sub_matrix, middle_row_index, label_flag, save_dir, chr_name, res):
    sub_matrix_str = f"array({sub_matrix.tolist()})"  
    file_path = os.path.join(save_dir, f"{chr_name}_{res}_sub_matrix.txt")
    with open(file_path, 'a', encoding='utf-8') as f:  
        f.write(f"({sub_matrix_str}, {middle_row_index}, {label_flag})\n")


label_dir = "HyperDeepTAD/model/label/"
matrix_dir = "HyperDeepTAD/data_pre_processing/hypergraph/"
save_dir1 = "HyperDeepTAD/model/model_data/"

other1 = ".txt" 
other2 = "_VE_matrix.txt"


os.makedirs(save_dir1, exist_ok=True)



chr_list = [f"chr{i}" for i in range(1, 23)]

res_list = [25,50,100]


for i in chr_list:
    label_file = os.path.join(label_dir, f"{i}{other1}")
    labels = load_labels(label_file)

    for res in res_list:
        matrix_file = os.path.join(matrix_dir, f"{i}_{res}{other2}")

        sub_matrix_size = (11, 50)

        large_matrix = load_all_lines_as_matrix(matrix_file)


        split_matrix_by_sliding_window(large_matrix, sub_matrix_size, labels, save_dir1, i, res)

print("All chromosomes have been processed.")
