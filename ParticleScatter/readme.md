# Particle Scattering experiment (Table 1(b))

## Overview
In this subdirectory, we provide the implementations for our methods and baselines reported in the table 1(b) of our paper.

## Installation
Using ```pip```
```bash
sudo apt update
sudo apt install python3.8.18
git clone https://github.com/tiendatnguyen-vision/Orbit-symmetrize.git 
cd ParticleScatter
bash install.sh
```

## Running
```bash
cd exp_lorentz
bash lorentz_scalar_mlp.sh
bash lorentz_mlp.sh
bash lorentz_mlp_aug.sh
bash lorentz_canonical.sh
bash lorentz_ps.sh
```




