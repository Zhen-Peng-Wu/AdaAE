## Requirements

```shell
pytorch 1.8.1+cu111
pytorch_geometric 2.0.2
sklearn 1.0.1 
rdkit 2022.9.2
multiprocess 0.70.14
```

## Dataset Preparation
### TUDataset 

```shell
$ python download_dataset.py
```
### ChEMBL
```shell
$ cd ./data
$ wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
$ unzip chem_dataset.zip
$ rm -rf dataset/*/processed
$ mv dataset chem_data
```

## Quick Start

A quick start example is given by:
```shell
$ python auto_test.py --data_root ./data/ --learning_type unsupervised --dataset MUTAG
```

An example of auto search is as follows:
```shell
$ python auto_main.py --data_root ./data/ --learning_type unsupervised --dataset MUTAG
or
$ python auto_main.py --data_root ./data/ --learning_type unsupervised --dataset IMDB-BINARY
```
