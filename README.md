# HyperDeepTAD: A method for constructing hypergraphs based on high-order reads and combining deep learning models to identify TADs



# About HyperDeepTAD

High-order interaction information is preserved through hypergraph modeling, and node transition probabilities are calculated to quantify dynamic interactions among multiple segments; the transition probability matrix is input into a dynamic convolutional network to capture local features, which are then processed by BiGRU to capture long-distance dependencies, with residual connections strengthening feature associations; boundaries are optimized by combining hypergraph clustering coefficients, and hierarchical TADs are obtained using cosine similarity.



# Requirements

tensorflow-gpu=2.8.0 ，numpy =1.24.3  ，pandas= 2.0.3  ，scikit-learn=1.3.2 ，tqdm= 4.67.1  



# Usage

## First--prepare_data

High-level reading access number:GSE202539

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

3.Count frequencies and divide into subgraphs

```python
python chr.py
python sort.py
python gk.py
python all_gk.py
```

## Third --Model

```python
cd model
python model.py
#predicted
load_model.ipynb
```

## Final --Filtering false-positive borders and assembling merged TADs

```
cd screen_and_nest
python clustering_coefficient.py
python screen.py
python merge_TAD.py
```





