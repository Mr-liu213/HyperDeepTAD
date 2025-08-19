# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import os
from utils import get_config
from scipy.sparse import csr_matrix, lil_matrix

# Convert raw data to a sparse matrix
def make_sparse_matrix(raw_data, weights, m, n):
    indptr = [0]
    indices = []
    data = []
    for row, weight in zip(raw_data, weights):
        indices.extend(row)
        data.extend([weight] * len(row))
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape=(n, m), dtype='float32')

def save_dense_matrix(matrix, filename):
    dense_matrix = matrix.toarray().astype(int) 
    with open(filename, "w") as f:
        for row in dense_matrix[1:]:
            f.write(" ".join(map(str, row)) + "\n")
    f.close()

# Construct the hypergraph class
class HyperGraph():
    def __init__(self, is_weighted=True):  # Is there a weight (contact frequency)?
        self.is_weighted = is_weighted

    def build_graph(self, node_list, edge_list, weight): 
        self.nodes = node_list
        self.edges = edge_list  # the neighbors of hyperedges (without weight)

     
        n_edge = [[] for _ in range(int(np.max(node_list) + 1))]

    
        self.node_degree = np.zeros((int(np.max(node_list) + 1)))

   
        self.edge_degree = np.array([len(e) for e in self.edges])

        for i, (e, w) in enumerate(tqdm(zip(edge_list, weight), total=len(edge_list))): 
            if isinstance(e, tuple):  
                e = list(e)
            e.sort()  
            ww = w 
            for v in e: 
                n_edge[v].append((i, ww)) 
                self.node_degree[v] += 1  

        for v in tqdm(node_list):
            n_edge_i = sorted(n_edge[v]) 
            n_edge[v] = np.array(n_edge_i, dtype=object)  

        self.n_edge = n_edge  

        print('adj matrix:')

       
        self.EV = make_sparse_matrix(
            edge_list, weight, int(np.max(node_list) + 1), len(edge_list))
        self.delta = lil_matrix((self.EV.shape[1], self.EV.shape[1]))
        size = np.array([1 / np.sqrt(len(e)) for e in self.edges])
        self.delta.setdiag(size)
        self.EV_over_delta = self.EV * self.delta
        self.VE = self.EV.T
        self.VE_over_delta = self.delta * self.VE
        # print("EV:",self.EV)
        # print("EV size", self.VE.shape)


config = get_config()
resolution_list = config['resolution']
chromosome_list = config['chrom_list']
temp_dir1 = config['temp_dir1']
temp_dir5 = config['temp_dir5']
temp_dir6 = config['temp_dir6']
if not os.path.exists(temp_dir6):
	os.mkdir(temp_dir6)
for res in resolution_list:
    i = 0
    for chr in chromosome_list:
        # Create an instance object
        G = HyperGraph()
        file1 = os.path.join(temp_dir1, str(int(res / 1000)) + "KB_chrom_ranges.npy")
        print("file1:",file1)
        chrom_range = np.load(file1,allow_pickle=True)
        num = []
        for v in chrom_range:
            num.append(v[1] - v[0])

        # print(num)
        num_list = num[i]
        i += 1
        print(num_list)
        node_list = np.arange(num_list+1).astype('int')
        data_list = []
        file2 = os.path.join(temp_dir5,chr+"_"+str(int(res/1000))+"_all_counter.npy")
        print("file2:",file2)
        data = np.load(file2,allow_pickle=True)
        for datum in data:
            data_list.append(datum)

        edge_list = np.array(data_list)

        print("edge_list:",edge_list)
        file3 = os.path.join(temp_dir5,chr+"_"+str(int(res/1000))+"_all_freq_counter.npy")
        print("file3:",file3)
        weight = np.load(file3,allow_pickle=True)
        print("weight:",weight)
        G.build_graph(node_list,edge_list,weight)
        # print("G.VE:",G.VE)
        file4 = os.path.join(temp_dir6, chr +"_"+ str(int(res / 1000)) +"_VE_matrix.txt")
        print("file4:",file4)
        save_dense_matrix(G.VE, file4)
