import torch
import torch.nn as nn
import torch.optim as optim
from run_nerf_helpers_deform import img2mse

# 1. 定义 PatchGAN 判别器
class MouthDiscriminator(nn.Module):
    """
    PatchGAN Discriminator specifically for high-frequency mouth details.
    Input: (B, 3, 64, 64) -> Output: (B, 1, 6, 6)
    """
    def __init__(self, input_nc=3, ndf=64):
        super(MouthDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

def crop_mouth_region(images, landmarks, size=64):
    """
    辅助函数：根据关键点裁剪嘴部区域
    images: [B, H, W, 3] tensor
    landmarks: [B, 68, 2] 嘴部关键点索引通常为 48-68
    """
    # 模拟裁剪逻辑：计算嘴部中心并 Crop
    # 实际项目中应使用 roi_align 或 grid_sample 进行可微分裁剪
    B = images.shape[0]
    cropped = []
    for i in range(B):
        # 简化：假设已获得嘴部中心坐标 cx, cy
        cx, cy = 256, 300 # Mock center
        half = size // 2
        # 注意边界检查
        patch = images[i, cy-half:cy+half, cx-half:cx+half, :]
        cropped.append(patch.permute(2, 0, 1)) # -> [3, 64, 64]
    
    return torch.stack(cropped)

def gan_finetune_step(nerf_model, discriminator, batch_data, optim_G, optim_D, criterion_GAN):
    """
    单步对抗微调
    """
    real_img = batch_data['image'].cuda()
    aud_feat = batch_data['audio'].cuda()
    landmarks = batch_data['landmarks'].cuda()
    
    # --- 1. Generator Forward (NeRF Rendering) ---
    # 此处省略渲染参数细节，假设 forward_render 返回全图
    fake_img = nerf_model(aud_feat, ...) 
    
    # 裁剪嘴部区域
    real_mouth = crop_mouth_region(real_img, landmarks)
    fake_mouth = crop_mouth_region(fake_img, landmarks)
    
    # --- 2. Update Discriminator ---
    optim_D.zero_grad()
    
    # Real
    pred_real = discriminator(real_mouth)
    loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
    
    # Fake
    pred_fake = discriminator(fake_mouth.detach()) # Detach G
    loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
    
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    optim_D.step()
    
    # --- 3. Update Generator (NeRF) ---
    optim_G.zero_grad()
    
    # Pixel Loss (MSE)
    loss_mse = img2mse(fake_img, real_img)
    
    # Adversarial Loss (骗过 D)
    pred_fake_G = discriminator(fake_mouth)
    loss_G_adv = criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
    
    # 总损失：MSE 保证整体结构，Adv 提升纹理锐度
    lambda_adv = 0.01
    loss_G_total = loss_mse + lambda_adv * loss_G_adv
    
    loss_G_total.backward()
    optim_G.step()
    
    return loss_G_total.item(), loss_D.item()

# 示例：初始化与运行
if __name__ == "__main__":
    # 假设已加载预训练好的 NeRF 模型
    nerf_model = ... 
    discriminator = MouthDiscriminator().cuda()
    
    criterion_GAN = nn.BCELoss()
    optim_G = optim.Adam(nerf_model.parameters(), lr=1e-5)
    optim_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    print("Starting GAN Fine-tuning...")
    # 模拟循环
    # for epoch in range(10):
    #     g_loss, d_loss = gan_finetune_step(...)
    #     print(f"G_Loss: {g_loss}, D_Loss: {d_loss}")