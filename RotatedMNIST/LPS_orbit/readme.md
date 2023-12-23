## Overview
In this repository, we provide the implementation for the two models CNN-Canonical-Orbit and CNN-PS-Orbit reported in table 1(a). We trained our models on three randomly selected seeds 38, 42, 49. For each seed, we report the test performance using the sample size k among [1, 10, 20, 50] that leads to the best performance. We also provide the checkpoints for our trained models on these seeds in the directory "save".

## Running CNN-PS-Orbit (Table 1(a))

Train

```bash
python main.py --save_dir ps_orbit --seed 38
python main.py --save_dir ps_orbit --seed 42
python main.py --save_dir ps_orbit --seed 49
```

Test 

```bash
python inference.py --save_dir ps_orbit --seed 38 --inference_k 10
python inference.py --save_dir ps_orbit --seed 42 --inference_k 20
python inference.py --save_dir ps_orbit --seed 49 --inference_k 10
```

## Running CNN-Canonical-Orbit (Table 1(a))
Train

```bash
python main.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 38
python main.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 42
python main.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 49
```

Test 

```bash
python inference.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 38
python inference.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 42
python inference.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 49
```



