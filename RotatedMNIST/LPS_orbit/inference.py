import argparse
import os
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets import MnistRotDataset
from saving import find_ckpt_polylr, find_ckpt_onecyclelr
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.datasetup.datasets import split_dataset

from models.interface import InterfacedModel

parser = argparse.ArgumentParser()
# General setting 
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device_id", type=int, default=0)
# Dataset
parser.add_argument("--num_train", type=int, default=10000)
# Training
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--sample_size", type=int, default=1)
# Optimization
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--lr_decay", type=int, default=1000)
parser.add_argument("--lr_scheduler", type=str, default="onecycle", choices=["poly", "onecycle"])
parser.add_argument("--lr_pct", type=float, default=0.1)
parser.add_argument("--lr_div", type=float, default=1)
parser.add_argument("--lr_div_final", type=float, default=1000)
parser.add_argument("--b1", type=float, default=0.5, help="decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="decay of first order momentum of gradient")
parser.add_argument("--w_orbit", type=float, default=1.0, help="coefficient for the orbit distance loss")
parser.add_argument("--grad_clip", type=float, default=0.0, help="weight clipping magnitude")
# Architecture
parser.add_argument("--cnn_dropout", type=float, default=0.4)
parser.add_argument("--gen_m", type=int, default=200)
parser.add_argument("--gen_ch", type=int, default=64)
parser.add_argument("--gen_L", type=int, default=2)
parser.add_argument("--gen_threshold", type=float, default=0.2)
parser.add_argument("--gen_dz", type=int, default=10)
parser.add_argument("--gen_z_scale", type=float, default=1.0)
# Record
parser.add_argument("--save_dir", type=str, default="ps_orbit")
# Inference
parser.add_argument("--num_inference", type=int, default=1)
parser.add_argument("--inference_k", type=int, default=1)
parser.add_argument("--inference_id", type=int, default=0)
parser.add_argument("--epoch_id", type=int, default=-1)
parser.add_argument("--inference_device", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda:" + str(args.inference_device))

def inference_model(model, data_loader, criterion):
    model.eval()
    data_loader = tqdm(data_loader) 
    test_stats = {"num_samples": 0, "orbit_distance": 0, "loss_task": 0, "accuracy": 0}
    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            B = images.shape[0]
            logits, orbit_distance = model(images) 
            loss_task = criterion(logits, labels) # scalar
            accuracy = (torch.argmax(logits, dim=1) == labels).sum().item() / B 
            
            # Record
            test_stats["num_samples"] += B 
            test_stats["orbit_distance"] += B * orbit_distance.item()
            test_stats["loss_task"] += B * loss_task.item()
            test_stats["accuracy"] += B * accuracy
    test_stats["orbit_distance"] /= test_stats["num_samples"]
    test_stats["loss_task"] /= test_stats["num_samples"]
    test_stats["accuracy"] /= test_stats["num_samples"]
    
    return test_stats

if __name__ == "__main__":
    # Deterministic training 
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    
    model = InterfacedModel(seed=args.seed, m=args.gen_m, ch=args.gen_ch, num_layers=args.gen_L, noise_scale=args.gen_z_scale, dz=args.gen_dz, threshold=args.gen_threshold, dropout=args.cnn_dropout, sample_size=args.inference_k, device=device) 
    # load checkpoint
    if args.lr_scheduler == "poly":
        ckpt_path = find_ckpt_polylr(args)
    elif args.lr_scheduler == "onecycle":
        ckpt_path = find_ckpt_onecyclelr(args)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    # get dataset
    test_data = MnistRotDataset('MnistRot', mode="test")
    test_loader = LoaderTo(DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=False),device)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    test_stats = inference_model(model, test_loader, criterion)
    print("accuracy = ", test_stats["accuracy"])
    print("loss = ", test_stats["loss_task"])
    print("orbit distance = ", test_stats["orbit_distance"])
    
        