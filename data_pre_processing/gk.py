import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils import get_config
import itertools

#  Construct a dictionary of node combinations and count the frequency of combinations that meet the criteria

def build_combination_dict(cluster_size, node_batch, data_tensor, node_to_data_indices, min_frequency):
   
    combined_clusters = []
    cluster_frequencies = []
    
   
    for node_id in tqdm(node_batch):
        data_indices = node_to_data_indices[node_id]
        if tf.size(data_indices) == 0:
            continue
        node_data_list = tf.gather(data_tensor, data_indices)
        combination_counter = {}
        for data_entry in node_data_list:
            valid_elements = data_entry[data_entry > node_id]
            if tf.size(valid_elements) >= cluster_size - 1:
                element_combinations = generate_combinations(valid_elements, cluster_size - 1)
              
                for combination in element_combinations:
                    combination_tuple = tuple(combination.numpy())
                    if combination_tuple in combination_counter:
                        combination_counter[combination_tuple] += 1
                    else:
                        combination_counter[combination_tuple] = 1
        
        valid_combinations = {
            comb: freq for comb, freq in combination_counter.items() 
            if freq >= min_frequency
        }
        
        if valid_combinations:
            combination_keys = list(valid_combinations.keys())
            frequencies = np.array([valid_combinations[comb] for comb in combination_keys], dtype=np.int32)
            combination_keys = np.array(combination_keys, dtype=np.int32)

            combinations_with_node = tf.concat([
                tf.ones((len(combination_keys), 1), dtype=tf.int32) * node_id,
                tf.convert_to_tensor(combination_keys, dtype=tf.int32)
            ], axis=-1)
            
            combined_clusters.append(combinations_with_node)
            cluster_frequencies.append(frequencies)
    
    if combined_clusters:
        combined_clusters = tf.concat(combined_clusters, axis=0)
        cluster_frequencies = tf.concat([tf.convert_to_tensor(freq, dtype=tf.int32) for freq in cluster_frequencies], axis=0)
    else:
        combined_clusters = tf.zeros((0, cluster_size), dtype=tf.int32)
        cluster_frequencies = tf.zeros((0,), dtype=tf.int32)
    
    return node_batch, combined_clusters, cluster_frequencies


def generate_combinations(indices, combination_size):
    combos = list(itertools.combinations(indices.numpy(), combination_size))
    combinations = tf.experimental.numpy.array(np.array(combos))
    return combinations




config = get_config()
resolution_list = config['resolution']
chromosome_list = config['chrom_list']
max_cluster_size = config['max_cluster_size']
kmer_sizes = config['k-mer_size']
temp_directory1 = config['temp_dir1']
temp_directory3 = config['temp_dir3']
temp_directory4 = config['temp_dir4']
min_frequency_cutoff = tf.constant(config['min_freq_cutoff'], dtype=tf.int32)


if not os.path.exists(temp_directory4):
    os.makedirs(temp_directory4)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("使用GPU加速")
else:
    print("使用CPU计算")


for chromosome_index in range(len(chromosome_list)):
    for resolution in resolution_list:
        # load chromosome range data
        print(f"加载 {int(resolution/1000)}KB_chrom_ranges.npy")
        chromosome_range = np.load(os.path.join(temp_directory1, f"{int(resolution/1000)}KB_chrom_ranges.npy"))
        chromosome_range = chromosome_range[chromosome_index]
        node_count = np.max(chromosome_range) + 1
        chromosome_name = chromosome_list[chromosome_index]
        
        # load the sorted edge list data
        edge_list_path = os.path.join(temp_directory3, f"{chromosome_name}_{int(resolution/1000)}KB_sort_edge_lists.npy")
        cluster_data = np.load(edge_list_path, allow_pickle=True)
        
        for kmer_size in kmer_sizes:
            cluster_size = kmer_size
            valid_clusters = []
            for cluster in tqdm(cluster_data):
                if (len(cluster) >= cluster_size) and (len(cluster) <= max_cluster_size):
                    valid_clusters.append(np.array(cluster, dtype=np.int32))
            
            if not valid_clusters:
                continue
            cluster_data_tensor = tf.ragged.constant(valid_clusters, dtype=tf.int32)
            
            # Construct the mapping from nodes to data indices
            node_to_data_mapping = [[] for _ in range(node_count)]
            for data_index, cluster in enumerate(tqdm(valid_clusters)):
                for node in cluster:
                    node_to_data_mapping[node].append(data_index)
            
            node_to_data_indices = tf.ragged.constant(node_to_data_mapping, dtype=tf.int32)

            processed_clusters = []
            cluster_frequency_list = []
            
            # Process nodes in batches
            node_list = np.arange(node_count).astype('int')
            batch_size = 50
            node_batches = np.array_split(node_list, int(len(node_list) / batch_size))
            node_batches = iter(node_batches)
            remaining_batches = len(node_list)
            while remaining_batches > 0:
                for batch_nodes in node_batches:
                    batch_indices, batch_clusters, batch_frequencies = build_combination_dict(
                        cluster_size, batch_nodes, cluster_data_tensor, node_to_data_indices, 
                        min_frequency_cutoff
                    )
                    remaining_batches -= len(batch_indices)
                    
                    if tf.size(batch_clusters) > 0:
                        processed_clusters.append(batch_clusters.numpy())
                        cluster_frequency_list.append(batch_frequencies.numpy())
                    
                    print(f"剩余批次: {remaining_batches}")
            
            if processed_clusters:
                all_clusters = np.concatenate(processed_clusters, axis=0)
                all_frequencies = np.concatenate(cluster_frequency_list, axis=0)
                print()
                print(f"组合结果形状: {all_clusters.shape}")
                
                # Save the results
                output_path = os.path.join(
                    temp_directory4, 
                    f"{chromosome_name}_{int(resolution/1000)}_all_{cluster_size}_counter.npy"
                )
                frequency_path = os.path.join(
                    temp_directory4, 
                    f"{chromosome_name}_{int(resolution/1000)}_all_{cluster_size}_freq_counter.npy"
                )
                
                np.save(output_path, all_clusters)
                np.save(frequency_path, all_frequencies)