#!/usr/bin/env python3
import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='NeRF Rendering Script')
    parser.add_argument('--iters', type=str, default='500000_head.tar', help='Iteration checkpoint file')
    parser.add_argument('--names', type=str, default='french', help='Model name')
    parser.add_argument('--datasets', type=str, default='french', help='Dataset name')
    parser.add_argument('--aud', type=str, default='aud.npy', help='test audio deepspeech file')
    parser.add_argument('--near', type=float, default=0.5555068731307984, help='Near plane distance')
    parser.add_argument('--far', type=float, default=1.1555068731307983, help='Far plane distance')
    parser.add_argument('--bc_type', type=str, default='torso_imgs', help='Background type')
    parser.add_argument('--suffix', type=str, default='val', help='Experiment suffix')
    parser.add_argument('--render_factor', type=str, default='8', help='Render downsampling factor')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    path = f"dataset/finetune_models/{args.names}/{args.iters}"
    datapath = f"dataset/{args.datasets}/0"

    cmd = [
        "python", "NeRFs/run_nerf_deform.py",
        "--need_torso", "True",
        "--config", "dataset/test_config.txt",
        "--aud_file",args.aud,
        "--expname", f"{args.names}_{args.suffix}",
        "--expname_finetune", f"{args.names}_{args.suffix}",
        "--render_only",
        "--ft_path", path,
        "--datadir", datapath,
        "--bc_type", args.bc_type,
        "--near", str(args.near),
        "--far", str(args.far),
        '--render_test',
        '--render_factor', args.render_factor
    ]

    subprocess.run(cmd)