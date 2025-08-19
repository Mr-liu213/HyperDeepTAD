import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def hypergraph_feature_vector_random_walk(tad_nodes, hypergraph, walk_length=5, num_walks=10):
    valid_tad_nodes = [n for n in tad_nodes if 0 <= n < hypergraph.shape[0]]
    if not valid_tad_nodes:
        return np.zeros(hypergraph.shape[0])  
    visit_counts = np.zeros(hypergraph.shape[0])
    # Perform a random walk for each node
    for node in valid_tad_nodes:
        for _ in range(num_walks):
            current_node = node
            for _ in range(walk_length):
               
                hyperedges = np.where(hypergraph[current_node] > 0)[0]
                if len(hyperedges) == 0:
                    break
                selected_hyperedge = np.random.choice(hyperedges)
                
              
                nodes_in_hyperedge = np.where(hypergraph[:, selected_hyperedge] > 0)[0]
                if len(nodes_in_hyperedge) == 1: 
                    break
                next_node = np.random.choice([n for n in nodes_in_hyperedge if n != current_node])
                
           
                visit_counts[next_node] += 1
                current_node = next_node

    return visit_counts / (np.sum(visit_counts) + 1e-6) if np.sum(visit_counts) > 0 else visit_counts


def hypergraph_cosine_similarity(tad_a_nodes, tad_b_nodes, hypergraph):
    vec_a = hypergraph_feature_vector_random_walk(tad_a_nodes, hypergraph)
    vec_b = hypergraph_feature_vector_random_walk(tad_b_nodes, hypergraph)
    max_len = max(len(vec_a), len(vec_b))
    vec_a_padded = np.pad(vec_a, (0, max_len - len(vec_a)), mode='constant')
    vec_b_padded = np.pad(vec_b, (0, max_len - len(vec_b)), mode='constant')
    norm_a = np.linalg.norm(vec_a_padded)
    norm_b = np.linalg.norm(vec_b_padded)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    return np.dot(vec_a_padded, vec_b_padded) / (norm_a * norm_b)


def merge_adjacent_tads(initial_tads, hypergraph, cs_threshold):
    if not initial_tads:
        return []
    sorted_tads = sorted(initial_tads, key=lambda x: x['left'])
    merged = [sorted_tads[0]]
    
    for current in sorted_tads[1:]:
        last = merged[-1]
        last_nodes = [n for n in range(last['left'], last['right'] + 1) if n < hypergraph.shape[0]]
        current_nodes = [n for n in range(current['left'], current['right'] + 1) if n < hypergraph.shape[0]]
        cs = hypergraph_cosine_similarity(last_nodes, current_nodes, hypergraph)
        
        if cs >= cs_threshold:
            merged[-1] = {'left': last['left'], 'right': current['right']}
        else:
            merged.append(current)
    return merged


def get_all_nested_tads(initial_tads, hypergraph, cs_threshold_outer):
    all_levels = []
    current_level = [{'left': t['left'], 'right': t['right']} for t in initial_tads]
    
    while True:
        all_levels.append(current_level)
        merged_tads = merge_adjacent_tads(current_level, hypergraph, cs_threshold_outer)
        if len(merged_tads) == len(current_level):
            break
        current_level = merged_tads
    
    all_tads = []
    seen = set()
    for level in all_levels:
        for tad in level:
            key = (tad['left'], tad['right'])
            if key not in seen:
                seen.add(key)
                all_tads.append(tad)
    return sorted(all_tads, key=lambda x: (x['left'], x['right']))


def save_tads_simple(tads, output_path):
    with open(output_path, 'w') as f:
        for tad in tads:
            f.write(f"{tad['left']} {tad['right']}\n")
    print(f"TAD results have been saved to: {output_path}")



def read_boundaries_file(boundary_file):
    with open(boundary_file, 'r') as f:
        boundaries = [int(line.strip()) for line in f if line.strip()]
    return boundaries


def load_all_lines_as_matrix(file2):
    lines = []
    with open(file2, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append([int(value) for value in line.strip().split()])
    return np.array(lines)



def process_single_task(args):
   
    chromosome, resolution, base_output_dir = args
    try:
        print(f"\n【Process {os.getpid()}】Starting to process {chromosome} (resolution: {resolution}kb)...")
        

        boundary_file = f"HyperDeepTAD/screen_and_nest/boundary_only/{resolution}k/{chromosome}_best_boundaries_only.txt"
        hypergraph_file = f"HyperDeepTAD/data_pre_processing/hypergraph/{chromosome}_{resolution}_VE_matrix.txt"
        
   
        resolution_dir = os.path.join(base_output_dir, f"{resolution}_TAD")
        os.makedirs(resolution_dir, exist_ok=True)
        output_file = os.path.join(resolution_dir, f"{chromosome}.txt")
        
      
        if not os.path.exists(boundary_file):
            raise FileNotFoundError(f"Boundary file does not exist: {boundary_file}")
        if not os.path.exists(hypergraph_file):
            raise FileNotFoundError(f"Hypergraph file does not exist: {hypergraph_file}")
        
        boundaries = read_boundaries_file(boundary_file)
        print(f"[Process {os.getpid()}] {chromosome} {resolution}kb: Read {len(boundaries)} boundary points")
        hypergraph = load_all_lines_as_matrix(hypergraph_file)
  
        if len(boundaries) < 2:
            raise ValueError(f"Insufficient boundary points to generate initial TAD (only {len(boundaries)})）")
        initial_tads = [
            {'left': boundaries[i], 'right': boundaries[i+1]}
            for i in range(len(boundaries) - 1)
        ]
        print(f"[Process {os.getpid()}] {chromosome} {resolution}kb: Generated {len(initial_tads)} initial TADs")
        
        nested_tads = get_all_nested_tads(
            initial_tads=initial_tads,
            hypergraph=hypergraph,
            cs_threshold_outer=0.5,
        )
        
        save_tads_simple(nested_tads, output_file)
        print(f"[Process {os.getpid()}] {chromosome} {resolution}kb: Processing complete")
        return (chromosome, resolution, "success")
    
    except FileNotFoundError as e:
        print(f"[Process {os.getpid()}] {chromosome} {resolution}kb: {e}, skip")
        return (chromosome, resolution, f"File does not exist: {str(e)}")
    except Exception as e:
        print(f"[Process {os.getpid()}] {chromosome} {resolution}kb: Processing error: {str(e)}")
        return (chromosome, resolution, f"Error: {str(e)}")



if __name__ == "__main__":

    chromosomes = ['chr20', 'chr21', 'chr22']
    # chromosomes = [f'chr{i}' for i in range(1, 23)]
    resolutions = [25, 50, 100]
    base_output_dir = "TAD_results"
    num_workers = 4  
    
 
    os.makedirs(base_output_dir, exist_ok=True)

    tasks = [(chrom, res, base_output_dir) for chrom in chromosomes for res in resolutions]
    print(f"Total number of tasks: {len(tasks)}, processed in parallel using {num_workers} processes.")
    
  
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
     
        futures = {executor.submit(process_single_task, task): task for task in tasks}
    
        results = []
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append((task[0], task[1], f"Task submission failed: {str(e)}"))
    
 
    print("\n===== All tasks completed =====")
    for chrom, res, status in results:
        print(f"{chrom} {res}kb：{status}")