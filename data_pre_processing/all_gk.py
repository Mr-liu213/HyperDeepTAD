# -*- coding: utf-8 -*-
import numpy as np
import os
from utils import get_config



config = get_config()
resolution_list = config['resolution']
chromosome_list = config['chrom_list']
temp_dir3 = config['temp_dir3']
temp_dir4 = config['temp_dir4']
temp_dir5 = config['temp_dir5']
if not os.path.exists(temp_dir5):
	os.mkdir(temp_dir5)
count_list = config['k-mer_size']
for chr in chromosome_list:
    for res in resolution_list:
        result1 = []
        result2 = []
        file1 = os.path.join(temp_dir3, chr + "_" + str(int(res / 1000)) +"KB_sort_edge_lists.npy")
        data1 = np.load(file1,allow_pickle=True)
        data1_sub = {tuple(sublist) for sublist in data1}
        print("file1:",file1)
        print("len(data1):",len(data1))

        for i in count_list:
            file2 = os.path.join(temp_dir4, chr + "_" + str(int(res / 1000)) +"_all_" + str(i) + "_counter.npy")          
            file3 = os.path.join(temp_dir4, chr + "_" + str(int(res / 1000)) +"_all_" + str(i) + "_freq_counter.npy")           
            if not os.path.exists(file2) or not os.path.exists(file3):
                print(f"File missing: {file2} or {file3}, skipping...")
                continue

            data2 = np.load(file2,allow_pickle=True)
            print("file2:",file2)
            print("len(data2):",len(data2))

            data3 = np.load(file3,allow_pickle=True)
            print("file3:",file3)
            print("len(data3):",len(data3))
            for t,l in zip(data2,data3):
                 if tuple(t) in data1_sub:
                      result1.append(t)
                      result2.append(l)
            print(len(result1))
            print(sum(result2))
        print("*****************")
        print(len(result1))
        print(sum(result2))
        np.save(os.path.join(temp_dir5,chr+"_"+str(int(res/1000))+"_all_counter.npy") ,np.array(result1,dtype=object))
        np.save(os.path.join(temp_dir5,chr+"_"+str(int(res/1000))+"_all_freq_counter.npy") , np.array(result2,dtype=object))

            



     