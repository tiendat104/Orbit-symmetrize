# Rotated MNIST experiment (Table 1)

## Overview
In this subdirectory, we provide the implementations for our method and baselines reported in the table 1 of our paper.
Implementation for the models CNN-PS-Orbit and CNN-Canonical-Orbit can be found in the subdirectory **LPS_orbit**. Implementation for the baselines CNN-PS and CNN-Canonical can be found in the subdirectory **LPS**. You can find more detailed explaination in each of these subdirectories. 

## Installation
```bash
pytorch = 2.0.0+cu117 
h5py = 3.10.0
pytest = 7.4.3
optax = 0.1.7 
tqdm = 4.66.1
kornia = 0.7.0
```
Additionally, please install the following:
```bash
pip install git+https://github.com/mfinzi/olive-oil-ml
```





























