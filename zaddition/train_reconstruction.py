import os
import torch
import numpy as np
import imageio
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 假设这些模块在同一目录下，或者你已经配置好了 PYTHONPATH
# 引用原项目文件中的函数和类
from run_nerf_helpers_deform import img2mse, mse2psnr, to8b, create_nerf
from run_nerf_deform import render_dynamic_face_new

class NeRFDataset(Dataset):
    """
    重构的动态数据加载器
    功能：根据训练配置文件，动态加载图像、Pose，并进行视听特征对齐
    """
    def __init__(self, basedir, split='train', win_size=16):
        self.basedir = basedir
        self.split = split
        self.win_size = win_size
        
        # 加载 json 配置
        json_path = os.path.join(basedir, f'transforms_{split}.json')
        with open(json_path, 'r') as fp:
            self.meta = json.load(fp)
            
        # 加载音频特征 (DeepSpeech Features)
        self.aud_features = np.load(os.path.join(basedir, 'aud.npy'))
        
        # 预加载图像路径
        self.frames = self.meta['frames']
        print(f"[{split.upper()}] Dataset loaded: {len(self.frames)} frames.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # 1. 加载图像 (GT)
        img_path = os.path.join(self.basedir, 'com_imgs', str(frame['img_id']) + '.jpg')
        img = imageio.imread(img_path)
        img = (np.array(img) / 255.0).astype(np.float32)
        
        # 2. 获取相机 Pose
        pose = np.array(frame['transform_matrix']).astype(np.float32)
        
        # 3. 视听对齐 (Audio-Visual Alignment)
        # 动态截取以当前帧为中心的音频窗口
        aud_id = frame['aud_id']
        left_i = aud_id - self.win_size // 2
        right_i = aud_id + self.win_size // 2
        
        # 处理边界 padding
        pad_left = -left_i if left_i < 0 else 0
        pad_right = right_i - self.aud_features.shape[0] if right_i > self.aud_features.shape[0] else 0
        
        left_i = max(0, left_i)
        right_i = min(self.aud_features.shape[0], right_i)
        
        aud_win = self.aud_features[left_i:right_i]
        
        if pad_left > 0:
            aud_win = np.pad(aud_win, ((pad_left, 0), (0, 0)), mode='constant')
        if pad_right > 0:
            aud_win = np.pad(aud_win, ((0, pad_right), (0, 0)), mode='constant')
            
        return {
            'image': torch.FloatTensor(img),
            'pose': torch.FloatTensor(pose),
            'audio': torch.FloatTensor(aud_win),
            'intrinsics': self.meta # 简化处理，实际应返回 tensor
        }

def train_pipeline(args):
    # 1. 初始化数据
    train_dataset = NeRFDataset(args.datadir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # 2. 初始化模型 (复用原项目 create_nerf 接口)
    # 注意：这里需要传入 args，假设 args 包含了 create_nerf 所需的所有参数
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, \
    learned_codes, AudNet_state, optimizer_aud_state, \
    AudAttNet_state, optimizer_audatt_state, models = create_nerf(args)
    
    print("Training pipeline initialized. Starting optimization loop...")
    
    # 3. 训练循环
    for step in range(start, args.N_iters):
        for batch in train_loader:
            target_rgb = batch['image'].cuda().reshape(-1, 3) # Flatten
            pose = batch['pose'].cuda()
            aud = batch['audio'].cuda()
            
            # 随机光线采样 (Ray Batching)
            # 此处简化：假设已经有了 get_rays 函数生成 rays_o, rays_d
            # 在完整代码中需要调用 run_nerf_helpers_deform.get_rays
            H, W = 512, 512 # 假设分辨率
            focal = float(batch['intrinsics']['focal_len'])
            
            # ... (此处省略 rays 生成代码，直接模拟 batch_rays) ...
            # 模拟随机采样 1024 条光线
            batch_rays = torch.randn(args.N_rand, 3, 6).cuda() # Mock data
            target_s = target_rgb[:args.N_rand] # Mock target

            # 4. 前向传播 (Forward Pass)
            # 调用 DFRF 核心渲染函数
            rgb, disp, acc, _, extras, loss_trans = render_dynamic_face_new(
                H, W, focal, cx=W/2, cy=H/2, chunk=args.chunk, 
                rays=batch_rays,
                aud_para=aud,
                verbose=False, retraw=True,
                **render_kwargs_train
            )

            # 5. 计算损失 (Loss Computation)
            # (A) 重建损失
            img_loss = img2mse(rgb, target_s)
            
            # (B) 扭曲正则化损失 (复现论文公式 6 和 9)
            # 正则化：抑制非人脸区域的过度变形
            # extras['raw'][..., -1] 通常是 density (sigma)
            density = extras['raw'][..., -1].detach() 
            # loss_trans 是 Face Warping 模块返回的 offset loss
            loss_reg = torch.mean((1.0 - torch.sigmoid(density)) * loss_trans)

            total_loss = img_loss + args.lambda_reg * loss_reg

            # 6. 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                psnr = mse2psnr(img_loss)
                print(f"[Iter {step}] Loss: {total_loss.item():.4f}, PSNR: {psnr.item():.2f}")
                
            if step >= args.N_iters:
                break
        
        # 保存模型
        if step % args.i_save == 0:
            path = os.path.join(args.basedir, args.expname, '{:06d}.tar'.format(step))
            torch.save({
                'global_step': step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--basedir', type=str, default='./logs')
    parser.add_argument('--expname', type=str, default='demo_train')
    parser.add_argument('--N_iters', type=int, default=200000)
    parser.add_argument('--N_rand', type=int, default=1024)
    parser.add_argument('--lambda_reg', type=float, default=5e-8, help='Regularization weight')
    parser.add_argument('--i_save', type=int, default=10000)
    
    # ... 其他 DFRF 所需参数需在此补全 ...
    args, unknown = parser.parse_known_args()
    
    train_pipeline(args)