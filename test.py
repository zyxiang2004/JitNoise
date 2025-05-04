# 主目录定义
MAIN_PROGRESS_DIR = "all_progress_frames"  # 主目录名称
COMPOSITE_DIR = "composite_frames"  # 合成帧目录

import jittor as jt
from jittor import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import shutil

# 清空临时文件夹
if os.path.exists('progress_frames'):
    shutil.rmtree('progress_frames')
os.makedirs('progress_frames', exist_ok=True)


# 重新定义UNet结构（需与训练时完全一致）
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm(out_ch),
                nn.ReLU(),
                nn.Conv(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm(out_ch),
                nn.ReLU()
            )

        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.up1 = double_conv(256 + 128, 128)
        self.up2 = double_conv(128 + 64, 64)
        self.up3 = nn.Conv(64, out_channels, 1)
        self.pool = nn.Pool(2, op='maximum')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def execute(self, x, t):
        # 确保t是浮点型张量并添加维度
        t = t.reshape(-1, 1).float()  # 关键修复：将标量t转换为张量
        t_emb = self.time_emb(t).reshape(-1, 64, 1, 1)
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x = self.upsample(x3)
        x = jt.concat([x, x2], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = jt.concat([x, x1], dim=1)
        x = self.up2(x)
        x = self.up3(x)
        return x


# 扩散模型类（调整sample_step处理）
class DiffusionModel:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = jt.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = jt.cumprod(self.alpha, dim=0)

    def sample_step(self, model, xt, t):
        model.eval()
        with jt.no_grad():
            # 将t转换为张量并确保类型正确
            t_tensor = jt.array(t, dtype=jt.float32)
            beta_t = self.beta[t].reshape(-1, 1, 1, 1)
            alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
            # 传递时间步张量
            noise_pred = model(xt, t_tensor / self.T)
            x_prev = (xt - beta_t / jt.sqrt(1 - alpha_bar_t) * noise_pred) / jt.sqrt(alpha_t)
        if t > 0:
            x_prev += jt.sqrt(beta_t) * jt.randn_like(xt)
        return x_prev


# 修改后的图像保存函数（直接使用PIL）
def save_single_image(image, filename):
    """保存单张图片用于制作GIF"""
    image = (image.clamp(-1, 1) + 1) / 2
    image = image.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(filename)


# 新增GIF生成函数
def make_gif(frame_folder, output_gif, duration=50):
    frames = [Image.open(os.path.join(frame_folder, f))
              for f in sorted(os.listdir(frame_folder)) if f.endswith('.png')]
    frames[0].save(output_gif, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration, loop=0)


# 新增：拼接5x5画布的函数
def create_composite_frame(frame_paths, output_path, grid_size=(5, 5), frame_size=32):
    canvas = Image.new('RGB', (frame_size * grid_size[0], frame_size * grid_size[1]))
    for idx, path in enumerate(frame_paths):
        img = Image.open(path)
        row = idx // grid_size[0]
        col = idx % grid_size[0]
        canvas.paste(img, (col * frame_size, row * frame_size))
    canvas.save(output_path)


# 加载模型和初始化
model = UNet()
model.load_state_dict(jt.load('jittor_cifar10_1000epochs.jt'))
model.eval()
diffusion = DiffusionModel(T=1000)
os.makedirs('composite_frames', exist_ok=True)

# 生成25个小GIF并收集它们的帧目录
num_samples = 25
save_interval = 50
gif_folders = []

for sample_idx in range(num_samples):
    frame_folder = os.path.join(MAIN_PROGRESS_DIR, f"progress_frames_{sample_idx}")
    gif_folders.append(frame_folder)

    # 清理并创建独立文件夹
    os.makedirs(MAIN_PROGRESS_DIR, exist_ok=True)  # <--- 新增主目录创建
    if os.path.exists(frame_folder):
        shutil.rmtree(frame_folder)
    os.makedirs(frame_folder, exist_ok=True)

    # 采样过程
    sample = jt.randn(1, 3, 32, 32)
    frame_count = 0

    with jt.no_grad():
        for t in reversed(range(diffusion.T)):
            sample = diffusion.sample_step(model, sample, t)
            if t % 100 == 0:
                jt.clean()
            if t % save_interval == 0 or t == 0:
                save_single_image(sample[0], f'{frame_folder}/frame_{frame_count:04d}.png')
                frame_count += 1
            print(f'\rGenerating {sample_idx + 1}/25 | Step: {t:04d}', end='')

    # 生成单个GIF
    make_gif(frame_folder, f'test_samples/single_{sample_idx}.gif', duration=100)

# 创建合成GIF
print("\nCreating composite GIF...")
max_frames = max([len(os.listdir(f)) for f in gif_folders])

for frame_num in range(max_frames):
    frame_paths = []
    for folder in gif_folders:
        frame_file = sorted([f for f in os.listdir(folder) if f.endswith('.png')])[frame_num]
        frame_paths.append(os.path.join(folder, frame_file))

    # 生成合成帧
    create_composite_frame(
        frame_paths,
        f'composite_frames/composite_{frame_num:04d}.png',
        grid_size=(5, 5),
        frame_size=32
    )
    print(f'\rCompositing frame {frame_num + 1}/{max_frames}', end='')

# 生成最终GIF
make_gif('composite_frames', 'test_samples/final_composite.gif', duration=100)

# 清理临时文件（可选）
# shutil.rmtree('composite_frames')
# for folder in gif_folders:
#     shutil.rmtree(folder)

print("\nComposite GIF created at test_samples/final_composite.gif")


def cleanup():
    if os.path.exists(MAIN_PROGRESS_DIR):
        shutil.rmtree(MAIN_PROGRESS_DIR)
    if os.path.exists(COMPOSITE_DIR):
        shutil.rmtree(COMPOSITE_DIR)


cleanup()