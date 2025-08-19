# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import sys, os
from utils import get_config

def build_node_dict():
    # load the chromosome size file
    chrom_sizes_df = pd.read_table(chrom_size, header=None, sep="\t")
    chrom_sizes_df.columns = ['chr', 'size']
    print(chrom_sizes_df)

    # create a node mapping dictionary
    for res in resolution_list:
        bin_to_node = {}
        chromosome_ranges = []
        
        for j, chromosome in enumerate(chromosome_list):
            # Obtain the chromosome size
            size_tensor = tf.constant(np.max(chrom_sizes_df['size'][chrom_sizes_df['chr'] == chromosome]), dtype=tf.float32)
            resolution_tensor = tf.constant(res, dtype=tf.float32)
            
           
            is_nan_size = tf.math.is_nan(size_tensor)
            is_nan_res = tf.math.is_nan(resolution_tensor)
            
            if tf.logical_or(is_nan_size, is_nan_res):
                continue
            
            # Calculate the maximum number of bins on the chromosome
            max_bins_chromosome = tf.cast(tf.math.ceil(size_tensor / resolution_tensor), tf.int32)
            
          
            node_counter = 1
            temp_range = [node_counter]
            
            # Generate the mapping from bins to nodes
            for i in tf.range(max_bins_chromosome):
                resolution_int = tf.cast(resolution_tensor, tf.int32)
                bin_start = i * resolution_int
                bin_key = f"{chromosome}:{bin_start.numpy()}"
                bin_to_node[bin_key] = node_counter
                node_counter += 1
            
            temp_range.append(node_counter)
            chromosome_ranges.append(temp_range)
        
       
        resolution_str = f"{int(res/1000)}KB"
        np.save(os.path.join(temp_directory1, f"{resolution_str}_chrom_ranges.npy"), chromosome_ranges)
        np.save(os.path.join(temp_directory1, f"{resolution_str}_bin_to_node.npy"), bin_to_node)

def parse_file():
    for chromosome in chromosome_list:
        for res in resolution_list:
            print(f"Processing chromosome: {chromosome}, resolution: {res}")
            
         
            resolution_str = f"{int(res/1000)}KB"
            bin_to_node = np.load(os.path.join(temp_directory1, f"{resolution_str}_bin_to_node.npy"), allow_pickle=True).item()
      
            with open(cluster_file_path, "r") as file1:
                line = file1.readline()
                line_counter = 0
                edge_lists = []
                
                while line:
                    site_list = line.strip().split(" ")[1:]
                    node_list = []
                    
                    if (len(site_list) < 2) or (len(site_list) > max_cluster_length):
                        line = file1.readline()
                        continue
                    for site_info in site_list:
                        try:
                            if ":" in site_info:
                                parts = site_info.split(":", 1)
                                if len(parts) == 2:
                                    chrom, bin_pos = parts
                                else:
                                    print(f"Skipping invalid data: {site_info}")
                                    continue
                            else:
                                print(f"Skipping invalid data: {site_info}")
                                continue
                        except:
                            print(f"Error processing: {site_info}")
                            raise EOFError
                        if chrom not in chromosome_list:
                            continue
                        
                        if chrom == chromosome:
                         
                            bin_pos_tensor = tf.constant(int(bin_pos), dtype=tf.int32)
                            resolution_tensor = tf.constant(res, dtype=tf.float32)
                            resolution_int = tf.cast(resolution_tensor, tf.int32)
                            
                            bin_floor = tf.math.floordiv(bin_pos_tensor, resolution_int)
                            bin_start = bin_floor * resolution_int
                            
                            bin_key = f"{chrom}:{bin_start.numpy()}"
                            node_id = bin_to_node.get(bin_key)
                            
                            if node_id is not None:
                                node_list.append(node_id)
                    
                    unique_nodes = list(set(node_list))
                    
                    if len(unique_nodes) > max_cluster_length:
                        line = file1.readline()
                        continue
                    
                    unique_nodes.sort()
                    line_counter += 1
                    
                    if line_counter % 10000 == 0:
                        sys.stdout.flush()
                    
                    if len(unique_nodes) > 1:
                        edge_lists.append(unique_nodes)
                    
                    line = file1.readline()
                
                np.save(os.path.join(temp_directory2, f"{chromosome}_{resolution_str}_edge_lists.npy"), np.array(edge_lists, dtype=object))


# load the configuration
config = get_config()
resolution_list = config['resolution']
chromosome_list = config['chrom_list']
temp_directory1 = config['temp_dir1']
temp_directory2 = config['temp_dir2']
cluster_file_path = config['cluster_path']
chrom_size = config['chrom_size']
max_cluster_length = config['max_cluster_size']



if not os.path.exists(temp_directory1):
    os.makedirs(temp_directory1)
if not os.path.exists(temp_directory2):
    os.makedirs(temp_directory2)


build_node_dict()
parse_file()
print("Processing completed!")    