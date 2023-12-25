import sys
import os
root_dir = os.getcwd() + '/'
sys.path.append(os.path.dirname(root_dir + "emlp-pytorch/"))

import argparse
import pathlib as Path 
from tqdm import tqdm
import random
import numpy as np

import torch 
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from oil.datasetup.datasets import split_dataset

from emlp_pytorch.datasets import ParticleInteraction
from emlp_pytorch.trainer.utils import LoaderTo
from models.interface import EquiLorentzNet
from saving import create_save_dir_polylr

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=0)
# data
parser.add_argument("--ntrain", type=int, default=10000)
# Training
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--sample_size", type=int, default=1)
# Optimization
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--wo", type=float, default=1, help="coefficient for the orbit distance loss")
parser.add_argument("--wb", type=float, default=1, help="coefficient for bound loss")
parser.add_argument("--bound", type=float, default=10.0)
parser.add_argument("--loss_type", type=float, default=2)
# architecture
parser.add_argument("--mlp_ch", type=int, default=384)
parser.add_argument("--mlp_L", type=int, default=3)
parser.add_argument("--generator", type=str, default="scalar", choices=["basic", "scalar"])
parser.add_argument("--gen_type", type=int, default=4)
parser.add_argument("--gen_act", type=str, default="leaky_relu")
parser.add_argument("--gen_bn", action="store_true")

parser.add_argument("--gen_ch", type=int, default=256)
parser.add_argument("--gen_L", type=int, default=2)
parser.add_argument("--gen_Zscale", type=float, default=0.1)
parser.add_argument("--gen_dz", type=int, default=1)
parser.add_argument("--last_act", type=str, default="identity")
# record
parser.add_argument("--val_interval", type=int, default=1)
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--skip_test", action="store_true", default=True)
parser.add_argument("--save_dir", type=str, default="lorentz")
# Others 
parser.add_argument("--tmp_mode", action="store_true")
args = parser.parse_args()

device = torch.device("cuda:" + str(args.device))

if args.tmp_mode:
    args.save_dir = "tmp"

log_dir, ckpt_dir = create_save_dir_polylr(args)
writer = SummaryWriter(log_dir=log_dir)

def train_model(model, data_loader, epoch, optimizer):
    model.train()
    data_loader = tqdm(data_loader)
    train_stats = {"num_samples": 0, "orbit_distance": 0, "loss_task": 0, "loss_bound": 0}
    for idx, (data, labels) in enumerate(data_loader):
        data = data.to(device) # [B, 16]
        labels = labels.to(device) # [B, 1]
        
        optimizer.zero_grad()
        B = data.shape[0]
        logits, orbit_distance, loss_bound = model(data) 
        # logits: [B, 1], orbit_distance: scalar 
        
        loss_task = torch.mean((logits - labels)**2) # scalar 
        total_loss = loss_task + args.wo * orbit_distance + args.wb * loss_bound
        # Update parameters 
        total_loss.backward()
        optimizer.step()
        
        # Record 
        data_loader.set_description(f"train: Epoch [{epoch}/{args.epochs}]")
        train_stats["num_samples"] += B 
        train_stats["orbit_distance"] += B * orbit_distance.item()
        train_stats["loss_task"] += B * loss_task.item()
        train_stats["loss_bound"] += B * loss_bound.item()
    
    train_stats["orbit_distance"] /= train_stats["num_samples"]
    train_stats["loss_task"] /= train_stats["num_samples"]
    train_stats["loss_bound"] /= train_stats["num_samples"]
    return train_stats

def test_model(model, data_loader, epoch):
    model.eval()
    data_loader = tqdm(data_loader)
    test_stats = {"num_samples": 0, "orbit_distance": 0, "loss_task": 0, "loss_bound": 0}
    with torch.no_grad():
        for idx, (data, labels) in enumerate(data_loader):
            data = data.to(device) # [B, 16]
            labels = labels.to(device) # [B, 1]
            
            B = data.shape[0]
            logits, orbit_distance, loss_bound = model(data) 
            # logits: [B, 1], orbit_distance: scalar
            loss_task = torch.mean((logits - labels)**2) # scalar 
            
            # Record
            data_loader.set_description(f"test:  Epoch [{epoch}/{args.epochs}]")
            test_stats["num_samples"] += B 
            test_stats["orbit_distance"] += B * orbit_distance.item()
            test_stats["loss_task"] += B * loss_task.item()
            test_stats["loss_bound"] += B * loss_bound.item()
            
    test_stats["orbit_distance"] /= test_stats["num_samples"]
    test_stats["loss_task"] /= test_stats["num_samples"]
    test_stats["loss_bound"] /= test_stats["num_samples"]
    return test_stats
                                            
if __name__ == "__main__":
    # Deterministic training 
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    
    with FixedNumpySeed(args.seed), FixedPytorchSeed(args.seed):
        base_dataset = ParticleInteraction(args.ntrain+6000)
        datasets = split_dataset(base_dataset, splits={'train': args.ntrain, 'val': 1000, 'test': 5000})
    
    train_data = datasets['train']
    val_data = datasets['val']
    test_data = datasets['test']

    train_loader = LoaderTo(DataLoader(train_data, batch_size=args.batch_size,
                                          shuffle=True,num_workers=0, pin_memory=True))
    val_loader = LoaderTo(DataLoader(val_data, batch_size=args.batch_size,
                                          shuffle=False,num_workers=0, pin_memory=True))
    test_loader = LoaderTo(DataLoader(test_data, batch_size=args.batch_size,
                                          shuffle=False,num_workers=0, pin_memory=True))
    
    model = EquiLorentzNet(args=args, device=device)
    
    # Optimization
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = PolynomialLR(optimizer, total_iters = len(train_loader)*args.epochs, power=1.0)
    
    best_val_loss = 1e8
    best_test_loss = 1e8
    for epoch in range(args.epochs):
        train_stats = train_model(model, train_loader, epoch, optimizer)
        # Record train
        writer.add_scalar("0train/0_loss_task", train_stats["loss_task"], epoch)
        writer.add_scalar("0train/1_orbit_distance", train_stats["orbit_distance"], epoch)
        writer.add_scalar("0train/2_loss_bound", train_stats["loss_bound"], epoch)
        lr_scheduler.step()
        
        if epoch % args.val_interval == 0:
            val_stats = test_model(model, val_loader, epoch)
            if not args.skip_test:
                test_stats = test_model(model, test_loader, epoch)

            if val_stats["loss_task"] < best_val_loss:
                best_val_loss = val_stats["loss_task"]
                if not args.skip_test:
                    best_test_loss = test_stats["loss_task"]
                
                best_ckpt = ckpt_dir / "best.pt"
                torch.save(model.state_dict(), best_ckpt)
            
            # Record val & test
            writer.add_scalar("1val/0_loss_task", val_stats["loss_task"], epoch)
            writer.add_scalar("1val/1_orbit_distance", val_stats["orbit_distance"], epoch)
            writer.add_scalar("1val/2_loss_bound", val_stats["loss_bound"], epoch)
            
            if not args.skip_test:
                writer.add_scalar("2test/0_loss_task", test_stats["loss_task"], epoch)
                writer.add_scalar("2test/1_orbit_distance", test_stats["orbit_distance"], epoch)
                writer.add_scalar("2test/2_loss_bound", test_stats["loss_bound"], epoch)
        
        # save checkpoints
        if epoch % args.save_interval == 0:
            ckpt_path = ckpt_dir / "epoch{}.pt".format(epoch)
            torch.save(model.state_dict(), ckpt_path)
        last_ckpt_path = ckpt_dir / "last.pt"
        torch.save(model.state_dict(), last_ckpt_path)
        
