# HyperDeepTAD: A Deep Learning Method for Detecting Topologically Associated Domains from Multiplex Chromatin Interaction Data



# About HyperDeepTAD

High-order interaction information is preserved through hypergraph modeling, and node transition probabilities are calculated to quantify dynamic interactions among multiple segments; the transition probability matrix is input into a dynamic convolutional network to capture local features, which are then processed by BiGRU to capture long-distance dependencies, with residual connections strengthening feature associations; boundaries are optimized by combining hypergraph clustering coefficients, and hierarchical TADs are obtained using cosine similarity.



# Requirements

tensorflow-gpu=2.8.0 ，numpy =1.24.3  ，pandas= 2.0.3  ，scikit-learn=1.3.2 ，tqdm= 4.67.1  



# Usage

## First--prepare_data

High-level reading access number:GSE202539  GSE114242

Corresponding Hi-C data：https://zenodo.org/records/10822184

## Second--Building  hypergraph

1.Unzip the directory and enter it.

```python
cd data_pre_processing
```

2.Perform screening operations on high-order readings

```python
python bin.py
```

3.The mapping relationship from bins to nodes is constructed based on chromosome sizes and specified resolutions, files for chromosome node ranges and bin-node mappings are generated, and a foundational node numbering system is provided for subsequent hypergraph construction.

```python
python chr.py  
```

4. Edge list data for each chromosome and each resolution is read, the interior of each sublist is first sorted, followed by sorting the sublists by their first element, and standardized files of sorted edge lists are output.

```python
python sort.py
```

5.Based on sorted edge lists, node combinations are mined and their frequencies are counted according to the specified k-mer size, valid node combinations meeting the minimum frequency threshold are filtered out, and hyperedge candidate sets with frequency information are generated.

```python
python gk.py
```

6.Hyperedge candidate sets across different k-mer sizes are integrated, intersection filtering is performed with the original edge lists, and the final hyperedge lists and corresponding frequency weight files are output.

```python
python all_gk.py
```

7.Hyperedge lists and weight data are loaded to construct a hypergraph, core structures including the hypergraph's adjacency matrix (EV/VE matrices) are calculated, and the VE matrix is converted to a dense format and saved as a text file.

```python
python bg.py  
```

8.Label the matrix with corresponding tags

```python
cd ..
cd model
python label.py
```

## Third --Model

1.Model training

```python
python model.py
```

2. Perform prediction on the results

```python
#predicted
load_model.ipynb
```



## Final --Filtering false-positive borders and assembling merged TADs

1. Clustering coefficient calculation

```python
cd screen_and_nest
python clustering_coefficient.py
```

2.The calculated clustering coefficients are filtered

```python
python screen.py
```

 3.Nested TADs are obtained

```python
python merge_TAD.py
```



## Instruction

For the pre-input processing of data for the model, you can either follow the steps above to operate step by step, or implement all steps at once using the script run.ipynb, which contains the input and output of each module.

