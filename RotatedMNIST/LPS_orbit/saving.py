import os
from pathlib import Path

def create_save_dir_polylr(args):
    data_dir = "train{}-val{}-test{}-polylr".format(args.num_train, 12000-args.num_train, 50000)
    config_str = "seed{}-B{}-lr{}-wo{}-drop{}-clip{}-".format(args.seed, args.batch_size, args.lr, args.w_orbit, args.cnn_dropout, args.grad_clip)
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
    data_dir = "train{}-val{}-test{}-polylr".format(args.num_train, 12000-args.num_train, 50000)
    config_str = "seed{}-B{}-lr{}-wo{}-drop{}-clip{}-".format(args.seed, args.batch_size, args.lr, args.w_orbit, args.cnn_dropout, args.grad_clip)
    config_str += "k{}-m{}-ch{}-L{}-t{}-dz{}-Z{}".format(args.sample_size, args.gen_m, args.gen_ch, args.gen_L, args.gen_threshold, args.gen_dz, args.gen_z_scale)
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(args.inference_id)
    if args.epoch_id == -1:
        ckpt_file = "best.pt"
    else:
        ckpt_file = "epoch{}.pt".format(args.epoch_id)
    ckpt_path = ckpt_dir / ckpt_file
    print("ckpt_path = ", ckpt_path)
    return ckpt_path

def create_save_dir_onecylelr(args):
    data_dir = "train{}-val{}-test{}-onecyclelr".format(args.num_train, 12000-args.num_train, 50000)
    config_str = "seed{}-B{}-wo{}-drop{}-clip{}-".format(args.seed, args.batch_size, args.w_orbit, args.cnn_dropout, args.grad_clip)
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
    data_dir = "train{}-val{}-test{}-onecyclelr".format(args.num_train, 12000-args.num_train, 50000)
    config_str = "seed{}-B{}-wo{}-drop{}-clip{}-".format(args.seed, args.batch_size, args.w_orbit, args.cnn_dropout, args.grad_clip)
    config_str += "lr:{}-pct{}-div{}:{}-decay{}-".format(args.lr, args.lr_pct, args.lr_div, args.lr_div_final, args.lr_decay)
    config_str += "k{}-m{}-ch{}-L{}-t{}-dz{}-Z{}".format(args.sample_size, args.gen_m, args.gen_ch, args.gen_L, args.gen_threshold, args.gen_dz, args.gen_z_scale)
    
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(args.inference_id)
    if args.epoch_id == -1:
        ckpt_file = "best.pt"
    else:
        ckpt_file = "epoch{}.pt".format(args.epoch_id)
    ckpt_path = ckpt_dir / ckpt_file
    print("ckpt_path = ", ckpt_path)
    return ckpt_path
    
