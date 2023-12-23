import argparse
import os
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets import MnistRotDataset
from saving import create_save_dir_polylr, create_save_dir_onecylelr
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.datasetup.datasets import split_dataset

from models.interface import InterfacedModel

parser = argparse.ArgumentParser()
# General setting 
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device_id", type=int, default=0)
# Dataset
# Training
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--sample_size", type=int, default=1)
parser.add_argument("--patience", type=int, default=400)
# Optimization
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--lr_decay", type=int, default=1000)
parser.add_argument("--lr_scheduler", type=str, default="onecycle", choices=["poly", "onecycle"])
parser.add_argument("--lr_pct", type=float, default=0.1)
parser.add_argument("--lr_div", type=float, default=1.0)
parser.add_argument("--lr_div_final", type=float, default=1000.0)
parser.add_argument("--b1", type=float, default=0.5, help="decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="decay of first order momentum of gradient")
# Architecture
parser.add_argument("--cnn_dropout", type=float, default=0.4)
parser.add_argument("--gen_m", type=int, default=200)
parser.add_argument("--gen_ch", type=int, default=64)
parser.add_argument("--gen_L", type=int, default=2)
parser.add_argument("--gen_threshold", type=float, default=0.2)
parser.add_argument("--gen_dz", type=int, default=10)
parser.add_argument("--gen_z_scale", type=float, default=1.0)
# Record
parser.add_argument("--save_dir", type=str, default="lps")
parser.add_argument("--skip_test", action="store_true", default=True)
parser.add_argument("--val_interval", type=int, default=1)
parser.add_argument("--save_interval", type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda:" + str(args.device_id))

if args.lr_scheduler == "poly":
    log_dir, ckpt_dir = create_save_dir_polylr(args)
elif args.lr_scheduler == "onecycle":
    log_dir, ckpt_dir = create_save_dir_onecylelr(args)
writer = SummaryWriter(log_dir = log_dir)

def train_model(model, data_loader, epoch, optimizer, criterion): # checked
    model.train()
    data_loader = tqdm(data_loader) 
    train_stats = {"num_samples": 0, "loss_task": 0, "accuracy": 0}
    for idx, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        B = images.shape[0]
        logits = model(images) # [B, 10]
        loss_task = criterion(logits, labels) # scalar
        accuracy = (torch.argmax(logits, dim=1) == labels).sum().item() / B 
        
        loss_task.backward()
        optimizer.step()
        
        # Record 
        data_loader.set_description(f"train: Epoch [{epoch}/{args.epochs}]")
        train_stats["num_samples"] += B 
        train_stats["loss_task"] += B * loss_task.item()
        train_stats["accuracy"] += B * accuracy 
    
    train_stats["loss_task"] /= train_stats["num_samples"]
    train_stats["accuracy"] /= train_stats["num_samples"]
    return train_stats 

def test_model(model, data_loader, criterion, epoch):
    model.eval()
    data_loader = tqdm(data_loader) 
    test_stats = {"num_samples": 0, "loss_task": 0, "accuracy": 0}
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            B = images.shape[0]
            logits = model(images) # [B, 10]
            loss_task = criterion(logits, labels) # scalar
            accuracy = (torch.argmax(logits, dim=1) == labels).sum().item() / B 
            
            # Record
            data_loader.set_description(f"test:  Epoch [{epoch}/{args.epochs}]")
            test_stats["num_samples"] += B 
            test_stats["loss_task"] += B * loss_task.item()
            test_stats["accuracy"] += B * accuracy
        
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
    
    model = InterfacedModel(seed=args.seed, m=args.gen_m, ch=args.gen_ch, num_layers=args.gen_L, noise_scale=args.gen_z_scale, dz=args.gen_dz, threshold=args.gen_threshold, dropout=args.cnn_dropout, sample_size=args.sample_size, device=device) 
    
    # get dataset
    train_data = MnistRotDataset('MnistRot', mode="train")
    val_data = MnistRotDataset('MnistRot', mode="val")
    test_data = MnistRotDataset('MnistRot', mode="test")

    train_loader = LoaderTo(DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True),device)
    val_loader = LoaderTo(DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True),device)
    test_loader = LoaderTo(DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True),device)
    
    # Optimization
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    if args.lr_decay == 0:
        decay_steps = args.epochs
    else:
        decay_steps = args.lr_decay 

    if args.lr_scheduler == "poly":
        lr_scheduler = PolynomialLR(optimizer, total_iters = len(train_loader)*decay_steps, power=1.0)
    else: 
        lr_scheduler = OneCycleLR(optimizer, total_steps = len(train_loader)*decay_steps, max_lr=args.lr, pct_start=args.lr_pct, div_factor=args.lr_div, final_div_factor=args.lr_div_final, anneal_strategy="linear")
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    # Main stage
    patience_count = 0
    for epoch in range(args.epochs):
        train_stats = train_model(model, train_loader, epoch, optimizer, criterion)
        # Record
        writer.add_scalar("0train/0_loss_task", train_stats["loss_task"], epoch)
        writer.add_scalar("0train/1_accuracy", train_stats["accuracy"], epoch)
        
        if epoch % args.val_interval == 0:
            val_stats = test_model(model, val_loader, criterion, epoch)
            if not args.skip_test:
                test_stats = test_model(model, test_loader, criterion, epoch)
            
            if val_stats["accuracy"] > best_val_acc:
                best_val_acc = val_stats["accuracy"]
                
                if not args.skip_test:
                    best_test_acc = test_stats["accuracy"]
                
                best_ckpt = ckpt_dir / "best.pt"
                torch.save(model.state_dict(), best_ckpt)
                patience_count = 0
            else:
                patience_count += args.val_interval
            
            # Record val 
            writer.add_scalar("1val/0_loss_task", val_stats["loss_task"], epoch)
            writer.add_scalar("1val/1_accuracy", val_stats["accuracy"], epoch)
            writer.add_scalar("1val/2_best_acc", best_val_acc, epoch)
            
            # Record test 
            if not args.skip_test:
                writer.add_scalar("2test/0_loss_task", test_stats["loss_task"], epoch)
                writer.add_scalar("2test/1_accuracy", test_stats["accuracy"], epoch)
                writer.add_scalar("2test/2_best_accuracy", best_test_acc, epoch)
        lr_scheduler.step()
        
        # save checkpoints 
        if epoch % args.save_interval == 0:
            ckpt_path = ckpt_dir / "epoch{}.pt".format(epoch)
            torch.save(model.state_dict(), ckpt_path)
        last_ckpt_path = ckpt_dir / "last.pt"
        torch.save(model.state_dict(), last_ckpt_path)
        
        if patience_count > args.patience:
            print("Stop early!")
            break

            