# -*- coding: utf-8 -*-
import numpy as np
import os
from utils import get_config
config = get_config()

chromosome_list = config['chrom_list']
resolution_list = config['resolution']

temp_dir2 = config['temp_dir2']
temp_dir3 = config['temp_dir3']
if not os.path.exists(temp_dir3):
	os.mkdir(temp_dir3)

#Sorted
for chr in chromosome_list:
    for res in resolution_list:
        data = np.load(os.path.join(temp_dir2, f"{chr}_{int(res/1000)}KB_edge_lists.npy"),allow_pickle=True)
        # print("data:",data)


        sorted_data = np.array([sorted(sublist) for sublist in data], dtype=object)

        sorted_by_first_element = sorted(sorted_data, key=lambda x: (x[0], x[1]))



        output_file = os.path.join(temp_dir3,f"{chr}_{int(res/1000)}KB_sort_edge_lists.npy")
        
        np.save(output_file, np.array(sorted_by_first_element, dtype=object))
        
        print(f"Sorted data saved to {output_file}")

