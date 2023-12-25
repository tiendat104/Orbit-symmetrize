set -e
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install h5py==3.10.0 pytest==7.4.3 plum-dispatch==2.2.2 optax==0.1.7 tqdm==4.66.1
pip install git+https://github.com/mfinzi/olive-oil-ml
pip install kornia==0.7.0
pip install tensorboard==2.15.1