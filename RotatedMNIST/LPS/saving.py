import os
from pathlib import Path

def create_save_dir_polylr(args):
    data_dir = "train{}-val{}-test{}-polylr".format(10000, 2000, 50000)
    config_str = "seed{}-B{}-lr{}-drop{}-".format(args.seed, args.batch_size, args.lr, args.cnn_dropout)
    config_str += "k{}-m{}-ch{}-L{}-t{}-dz{}-Z{}".format(args.sample_size, args.gen_m, args.gen_ch, args.gen_L, args.gen_threshold, args.gen_dz, args.gen_z_scale)
    
    log_dir = Path("save/logs") / data_dir / args.save_dir / config_str 
    os.makedirs(log_dir, exist_ok=True)
    num_exist_running = len(os.listdir(log_dir))
    log_dir = log_dir / str(num_exist_running)
    os.makedirs(log_dir)
    
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(num_exist_running)
    os.makedirs(ckpt_dir)
    return log_dir, ckpt_dir

def find_ckpt_polylr(args):
    data_dir = "train{}-val{}-test{}-polylr".format(10000, 2000, 50000)
    config_str = "seed{}-B{}-lr{}-drop{}-".format(args.seed, args.batch_size, args.lr, args.cnn_dropout)
    config_str += "k{}-m{}-ch{}-L{}-t{}-dz{}-Z{}".format(args.sample_size, args.gen_m, args.gen_ch, args.gen_L, args.gen_threshold, args.gen_dz, args.gen_z_scale)
    
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(args.infer_id)
    if args.infer_epoch == -1:
        ckpt_file = "best.pt"
    else:
        ckpt_file = "epoch{}.pt".format(args.infer_epoch)
    ckpt_path = ckpt_dir / ckpt_file
    print("ckpt_path = ", ckpt_path)
    return ckpt_path

def create_save_dir_onecylelr(args):
    data_dir = "train{}-val{}-test{}-onecyclelr".format(10000, 2000, 50000)
    config_str = "seed{}-B{}-drop{}-".format(args.seed, args.batch_size, args.cnn_dropout)
    config_str += "lr:{}-pct{}-div{}:{}-decay{}-".format(args.lr, args.lr_pct, args.lr_div, args.lr_div_final, args.lr_decay)
    config_str += "k{}-m{}-ch{}-L{}-t{}-dz{}-Z{}".format(args.sample_size, args.gen_m, args.gen_ch, args.gen_L, args.gen_threshold, args.gen_dz, args.gen_z_scale)
    
    log_dir = Path("save/logs") / data_dir / args.save_dir / config_str 
    os.makedirs(log_dir, exist_ok=True)
    num_exist_running = len(os.listdir(log_dir))
    log_dir = log_dir / str(num_exist_running)
    os.makedirs(log_dir)
    
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(num_exist_running)
    os.makedirs(ckpt_dir)
    return log_dir, ckpt_dir

def find_ckpt_onecyclelr(args):
    data_dir = "train{}-val{}-test{}-onecyclelr".format(10000, 2000, 50000)
    config_str = "seed{}-B{}-drop{}-".format(args.seed, args.batch_size, args.cnn_dropout)
    config_str += "lr:{}-pct{}-div{}:{}-decay{}-".format(args.lr, args.lr_pct, args.lr_div, args.lr_div_final, args.lr_decay)
    config_str += "k{}-m{}-ch{}-L{}-t{}-dz{}-Z{}".format(args.sample_size, args.gen_m, args.gen_ch, args.gen_L, args.gen_threshold, args.gen_dz, args.gen_z_scale)
    
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(args.infer_id)
    if args.infer_epoch == -1:
        ckpt_file = "best.pt"
    else:
        ckpt_file = "epoch{}.pt".format(args.infer_epoch)
    ckpt_path = ckpt_dir / ckpt_file
    print("ckpt_path = ", ckpt_path)
    return ckpt_path





