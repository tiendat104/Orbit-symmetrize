import os 
from pathlib import Path

def create_save_dir_polylr(args):
    data_dir = "N{}".format(args.ntrain)
    
    config_str = "B{}-k{}-".format(args.batch_size, args.sample_size)
    config_str += "lr{}-bound{}-wo{}-wb{}-".format(args.lr, args.bound, args.wo, args.wb)
    config_str += "mlp:ch{}-L{}-".format(args.mlp_ch, args.mlp_L)
    config_str += "gen:T{}-ch{}-L{}".format(args.gen_type, args.gen_ch, args.gen_L)
    
    log_dir = Path("save/logs") / data_dir / args.save_dir / config_str 
    os.makedirs(log_dir, exist_ok=True)
    num_exist_running = len(os.listdir(log_dir))
    log_dir = log_dir / str(num_exist_running)
    os.makedirs(log_dir)
    
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(num_exist_running)
    os.makedirs(ckpt_dir)
    return log_dir, ckpt_dir

def find_ckpt_polylr(args): 
    data_dir = "N{}".format(args.ntrain)
    
    config_str = "B{}-k{}-".format(args.batch_size, args.sample_size)
    config_str += "lr{}-bound{}-wo{}-wb{}-".format(args.lr, args.bound, args.wo, args.wb)
    config_str += "mlp:ch{}-L{}-".format(args.mlp_ch, args.mlp_L)
    config_str += "gen:T{}-ch{}-L{}".format(args.gen_type, args.gen_ch, args.gen_L)
    
    ckpt_dir = Path("save/ckpts") / data_dir / args.save_dir / config_str / str(args.infer_id) 
    if args.epoch_id == -1:
        ckpt_file = "best.pt"
    else:
        ckpt_file = "epoch{}.pt".format(args.epoch_id)
    ckpt_path = ckpt_dir / ckpt_file
    print("ckpt_path = ", ckpt_path)
    return ckpt_path
