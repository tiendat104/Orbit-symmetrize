# Rotated MNIST experiment - (our method)

## Overview
In this repository, we provide the implementation for the two models CNN-Canonical-Orbit and CNN-PS-Orbit reported in table 1(a). We trained our models on three randomly selected seeds 38, 42, 49. For each seed, we report the test performance using the sample size k among [1, 10, 20, 50] that leads to the best performance.

## Running CNN-PS-Orbit (Table 1(a))

### Train

```bash
python main.py --save_dir ps_orbit --seed 38
python main.py --save_dir ps_orbit --seed 42
python main.py --save_dir ps_orbit --seed 49
```

### Inference

```bash
python inference.py --save_dir ps_orbit --inference_k 20 --seed 38
python inference.py --save_dir ps_orbit --inference_k 20 --seed 42
python inference.py --save_dir ps_orbit --inference_k 20 --seed 49
```

### Reproduce experiment results
inference_k in the above command corresponds to the sample size used in the interface. <br>
To replicate the test performance reported in the paper, please use following sample sizes for each seed:

| Seed | Sample size (k) |
|---|---|
| 38 | 10 |
| 42 | 20 |
| 49 | 10 |

For example, for the seed 38, please use the following command:
```bash
python inference.py --save_dir ps_orbit --inference_k 10 --seed 38
```

## Running CNN-Canonical-Orbit (Table 1(a))
### Train

```bash
python main.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 38
python main.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 42
python main.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 49
```

### Inference

```bash
python inference.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 38
python inference.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 42
python inference.py --gen_z_scale 0.0 --save_dir canon_orbit --seed 49
```

## Trained models
Trained model checkpoints can be found at [this link](https://drive.google.com/file/d/1kpups1xcSszySZnlkcLCk0D0bMgjQ3Jd/view?usp=sharing).
Please unzip the orbit.zip file and put the "save" directory inside this repository. Then, you can directly run the above inference commands on these checkpoints.

