import pandas as pd
import os

def filter_continuous_boundaries(input_path, output_path, boundary_only_path):
 
    df = pd.read_csv(input_path)
    required_columns = ['Boundary', 'upstream_cc', 'cross_boundary_cc', 'downstream_cc']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following columns are missing in the file: {missing_columns}")
  
    df['Boundary'] = df['Boundary'].astype(int)
    df['upstream_cc'] = df['upstream_cc'].astype(float)
    df['cross_boundary_cc'] = df['cross_boundary_cc'].astype(float)
    df['downstream_cc'] = df['downstream_cc'].astype(float)
    
    condition1 = (df['upstream_cc'] > df['cross_boundary_cc']) & (df['downstream_cc'] > df['cross_boundary_cc'])
    filtered = df[condition1].sort_values(by='Boundary').reset_index(drop=True)
    
    if filtered.empty:
        print(f"No boundary nodes meeting the conditions: {input_path}")
        return None

    groups = []
    current_group = [filtered.iloc[0]]
    
    for i in range(1, len(filtered)):
        if filtered.iloc[i]['Boundary'] == filtered.iloc[i-1]['Boundary'] + 1:
            current_group.append(filtered.iloc[i])
        else:
            groups.append(pd.DataFrame(current_group))
            current_group = [filtered.iloc[i]]
    groups.append(pd.DataFrame(current_group))
    
  
    best_boundaries = []
    for group in groups:
 
        group_sorted = group.sort_values(by='cross_boundary_cc', ascending=True).reset_index(drop=True)
        best_boundaries.append(group_sorted.iloc[0])
    
    result = pd.DataFrame(best_boundaries)[['Boundary', 'upstream_cc', 'cross_boundary_cc', 'downstream_cc']]
    result.to_csv(output_path, index=False)
    
    boundary_list = result['Boundary'].tolist()
    with open(boundary_only_path, 'w', encoding='utf-8') as f:
        for boundary in boundary_list:
            f.write(f"{int(boundary)}\n")
    print(f"Processing completed: {input_path}")
    print(f"Complete results saved to: {output_path}")
    print(f"Boundary IDs saved to: {boundary_only_path}")
    
    return result





if __name__ == "__main__":
    input_parent_dir = "results/"
    output_root = "filtered_results"
    boundary_root = "boundary_only"

    resolutions = [25, 50, 100]
    res_subdirs = [f"{res}k" for res in resolutions]  
    

    chromosomes = [20, 21, 22]
    
    for res, res_subdir in zip(resolutions, res_subdirs):
        print(f"\n===== Processing resolution: {res_subdir} =====")
        
        input_dir = os.path.join(input_parent_dir, res_subdir)
        if not os.path.exists(input_dir):
            print(f"Warning: Input directory for resolution {res_subdir} does not exist → {input_dir}, skipping this resolution")
            continue

        output_dir = os.path.join(output_root, res_subdir)
        boundary_dir = os.path.join(boundary_root, res_subdir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(boundary_dir, exist_ok=True)
        print(f"Output directory: {output_dir}, Boundary directory: {boundary_dir}")
        
        for chr_num in chromosomes:
            print(f"\n--- chr{chr_num} ---")
            
          
            input_path = os.path.join(input_dir, f"chr{chr_num}_clustering_results.txt")
          
            output_path = os.path.join(output_dir, f"chr{chr_num}_best_continuous_boundaries.txt")
            boundary_only_path = os.path.join(boundary_dir, f"chr{chr_num}_best_boundaries_only.txt")
            
          
            if not os.path.exists(input_path):
                print(f"Warning: Input file for chromosome chr{chr_num} does not exist → {input_path}, skipping")
                continue
            
          
            try:
                print("input_path:",input_path)
                print("output_path:",output_path)
                print("boundary_only_path:",boundary_only_path)
                filter_continuous_boundaries(input_path, output_path, boundary_only_path)
                print(f"Processing completed: {input_path} → output to {output_path} and {boundary_only_path}")
            except Exception as e:
                print(f"Error processing chromosome chr{chr_num}: {e}, skipping")
                continue








