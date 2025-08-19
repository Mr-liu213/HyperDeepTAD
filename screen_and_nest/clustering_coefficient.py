import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_neighbors(node_1based, incidence_matrix, direction=None):
    """
    Parameters:
        node_1based: Current node (1-based)
        direction: 'upstream' (smaller node number)、'downstream' (larger node number)
    """
    neighbors = set()
   
    node_idx = node_1based - 1  
    for j in range(incidence_matrix.shape[1]):
        if incidence_matrix[node_idx, j] > 0:  
            edge_nodes_0based = np.where(incidence_matrix[:, j] > 0)[0]
            edge_nodes_1based = [n + 1 for n in edge_nodes_0based]  
            if direction == 'upstream':
                edge_nodes_1based = [n for n in edge_nodes_1based if n < node_1based]
            elif direction == 'downstream':
                edge_nodes_1based = [n for n in edge_nodes_1based if n > node_1based]
            
            neighbors.update(edge_nodes_1based)
    
    neighbors.discard(node_1based)
    return sorted(neighbors)

def calculate_shared_edges_weight(node_list_1based, incidence_matrix):
    """Calculate the sum of weights of shared hyperedges for all node pairs in the 1-based node list"""
    shared_edges_weight = 0
   
    for i in range(len(node_list_1based)):
        for j in range(i + 1, len(node_list_1based)):
            n1 = node_list_1based[i]
            n2 = node_list_1based[j]
            n1_idx = n1 - 1
            n2_idx = n2 - 1
            
            common_edges = np.where((incidence_matrix[n1_idx] > 0) & (incidence_matrix[n2_idx] > 0))[0]
            for edge in common_edges:
                shared_edges_weight += min(incidence_matrix[n1_idx, edge], incidence_matrix[n2_idx, edge])
    
    return shared_edges_weight

def boundary_clustering_analysis(boundary_node_1based, incidence_matrix, max_distance=40):
    """
   Calculate the clustering coefficient of nodes
    """

    upstream_neighbors = get_neighbors(boundary_node_1based, incidence_matrix, direction='upstream')
    downstream_neighbors = get_neighbors(boundary_node_1based, incidence_matrix, direction='downstream')
    upstream_neighbors = [n for n in upstream_neighbors 
                         if abs(n - boundary_node_1based) <= max_distance]
    downstream_neighbors = [n for n in downstream_neighbors 
                          if abs(n - boundary_node_1based) <= max_distance]
    
    results = {}
    
    # 1. 上Upstream clustering coefficient

    if len(upstream_neighbors) >= 2:
        upstream_weight = calculate_shared_edges_weight(upstream_neighbors, incidence_matrix)
        k_up = len(upstream_neighbors)
        results['upstream_cc'] = (2 * upstream_weight) / (k_up * (k_up - 1))
    else:
        results['upstream_cc'] = 0
    
    # 2. Downstream clustering coefficient
    if len(downstream_neighbors) >= 2:
        downstream_weight = calculate_shared_edges_weight(downstream_neighbors, incidence_matrix)
        k_down = len(downstream_neighbors)
        results['downstream_cc'] = (2 * downstream_weight) / (k_down * (k_down - 1))
    else:
        results['downstream_cc'] = 0
    
    # 3. Downstream clustering coefficient
    if upstream_neighbors and downstream_neighbors:
      
        cross_nodes = upstream_neighbors + downstream_neighbors
        total_weight = calculate_shared_edges_weight(cross_nodes, incidence_matrix)
        
       
        upstream_internal = calculate_shared_edges_weight(upstream_neighbors, incidence_matrix) if len(upstream_neighbors)>=2 else 0
        downstream_internal = calculate_shared_edges_weight(downstream_neighbors, incidence_matrix) if len(downstream_neighbors)>=2 else 0
        cross_weight = total_weight - upstream_internal - downstream_internal
        
        max_possible = len(upstream_neighbors) * len(downstream_neighbors)
        results['cross_boundary_cc'] = cross_weight / max_possible if max_possible > 0 else 0
    else:
        results['cross_boundary_cc'] = 0
    
    return results



def load_all_lines_as_matrix(file2):
    lines = []
    with open(file2, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append([int(value) for value in line.strip().split()])
    matrix = np.array(lines)
    return matrix


def read_boundary_nodes(file_path):
   
    with open(file_path, 'r') as f:
        lines = f.readlines()
    boundaries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        boundaries.append(int(line))  
    return boundaries

def save_results_to_file(results, output_path):
    """
    Format: Boundary, Upstream clustering coefficient, Cross clustering coefficient, Downstream clustering coefficient

    """
    with open(output_path, 'w') as f:
 
        f.write("Boundary,upstream_cc,cross_boundary_cc,downstream_cc\n")

        for res in results:
            line = f"{res['boundary_node']},{res['upstream_cc']},{res['cross_boundary_cc']},{res['downstream_cc']}\n"
            f.write(line)
    print(f"Results have been saved to the file: {output_path}")


def load_all_lines_as_matrix(file2):
    lines = []
    with open(file2, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append([int(value) for value in line.strip().split()])
    return np.array(lines)


def read_boundary_nodes(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    boundaries = []
    for line in lines:
        line = line.strip()
        if line:
            boundaries.append(int(line))  
    return boundaries






def process_single_task(args):
   
    res, chr_num, base_matrix_path, base_boundary_prefix, output_dir, res_to_maxdistance = args
    try:
        pid = os.getpid()
        max_distance = res_to_maxdistance[res]
        res_tag = f"{res}k"
        print(f"[Process {pid}] Starting processing: resolution {res_tag}, chromosome chr{chr_num} maximum distance: {max_distance}")


        output_subdir = os.path.join(output_dir, res_tag)
        os.makedirs(output_subdir, exist_ok=True)
        
        boundary_res_dir = os.path.join(base_boundary_prefix, f"no_results_{res}")
        matrix_path = os.path.join(base_matrix_path, f"chr{chr_num}_{res}_VE_matrix.txt")
        boundary_path = os.path.join(boundary_res_dir, f"chr{chr_num}_boundary.txt")
        output_path = os.path.join(output_subdir, f"chr{chr_num}_clustering_results.txt")
     
        if not os.path.exists(boundary_res_dir):
            raise FileNotFoundError(f"Boundary directory does not exist: {boundary_res_dir}")
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file does not exist: {matrix_path}")
        if not os.path.exists(boundary_path):
            raise FileNotFoundError(f"Boundary file does not exist: {boundary_path}")
        
      
        incidence_matrix = load_all_lines_as_matrix(matrix_path)
        total_nodes = incidence_matrix.shape[0]
      
        boundary_nodes = read_boundary_nodes(boundary_path)
      

        valid_boundaries = [n for n in boundary_nodes if 1 <= n <= total_nodes]
        invalid_nodes = [n for n in boundary_nodes if not (1 <= n <= total_nodes)]
        if invalid_nodes:
            print(f"[Process {pid}] Warning: skipping invalid nodes: {invalid_nodes}")
        if not valid_boundaries:
            print(f"[Process {pid}] No valid boundary nodes, skipping")
            return (res, chr_num, "No valid boundary nodes")
        
        
        all_results = []
        
        for node in valid_boundaries:
            res1 = boundary_clustering_analysis(node, incidence_matrix, max_distance)
            result_dict = {
                'boundary_node': node,
                'upstream_cc': res1['upstream_cc'],
                'cross_boundary_cc': res1['cross_boundary_cc'],
                'downstream_cc': res1['downstream_cc']
            }
            all_results.append(result_dict)

    
        
        print(f"Completed calculation of clustering coefficients for chromosome {chr_num}, with a total of {len(all_results)} boundary nodes")
        
  
        save_results_to_file(all_results, output_path)
    
    except Exception as e:
        print(f"[Process {os.getpid()}] Error processing resolution {res}k chromosome {chr_num}: {e}")
        return (res, chr_num, f"Failed: {str(e)}")


def main():
   
    base_matrix_path = "HyperDeepTAD/data_pre_processing/hypergraph"
    base_boundary_prefix = "HyperDeepTAD/predicted"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    res_to_maxdistance = {25: 80, 50: 40, 100: 20}
 
    chromosomes = [20,21,22]
    resolutions = [25, 50, 100]
    max_outer_workers = 4  
    
 
    tasks = [
        (res, chr_num, base_matrix_path, base_boundary_prefix, output_dir, res_to_maxdistance)
        for res in resolutions
        for chr_num in chromosomes
    ]
    print(f"Total number of tasks: {len(tasks)}, running {max_outer_workers} tasks simultaneously")
 
    with ProcessPoolExecutor(max_workers=max_outer_workers) as outer_executor:
        futures = {outer_executor.submit(process_single_task, task): task for task in tasks}
  
        results = []
        for future in as_completed(futures):
            task = futures[future]
            res, chr_num = task[0], task[1]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((res, chr_num, f"Task submission failed: {str(e)}"))

    print("\n===== All tasks have been processed! =====")
    for res, chr_num, status in results:
        print(f"Resolution {res}k, chromosome {chr_num}: {status}")


if __name__ == "__main__":
    main()
