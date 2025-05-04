import jittor as jt
from jittor import nn, optim, dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging
import jittor.transform as transforms
# 设置设备
jt.flags.use_cuda = 1  # jt.has_cuda
jt.flags.amp_reg = 0  # 禁用自动混合精度（若未使用AMP）

# UNet 模型定义
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        def double_conv(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm(out_ch),
                nn.ReLU(),  # CBL
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
        t_emb = self.time_emb(t.reshape(-1, 1)).reshape(-1, 64, 1, 1)
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


# 扩散模型类
class DiffusionModel:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = jt.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = jt.cumprod(self.alpha, dim=0)

    def forward_process(self, x0, t, noise=None):
        """正向过程：添加噪声"""
        if noise is None:
            noise = jt.randn_like(x0)
        sqrt_alpha_bar = jt.sqrt(self.alpha_bar[t]).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = jt.sqrt(1 - self.alpha_bar[t]).reshape(-1, 1, 1, 1)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def sample_step(self, model, xt, t):
        """反向过程单步"""
        model.eval()
        with jt.no_grad():
            beta_t = self.beta[t].reshape(-1, 1, 1, 1)
            alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
            noise_pred = model(xt, t / self.T)
            x_prev = (xt - beta_t / jt.sqrt(1 - alpha_bar_t) * noise_pred) / jt.sqrt(alpha_t)
        if t > 0:
            x_prev += jt.sqrt(beta_t) * jt.randn_like(xt)
        jt.gc() #触发垃圾回收
        return x_prev


# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = dataset.CIFAR10(root='./data', train=True, download=True, transform=transform)
batch_size = 128
total_batches = (len(trainset) + batch_size - 1) // batch_size
trainloader = dataset.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# 训练设置
model = UNet()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
diffusion = DiffusionModel(T=1000)
num_epochs = 1000
os.makedirs('samples', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# 日志设置
logging.basicConfig(filename='logs/training.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
losses = []


# 保存图像的函数（Jittor 没有直接的 make_grid，需手动实现）
def save_image_grid(images, filename, nrow=4):
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    n = images.shape[0]
    ncols = nrow
    nrows = (n + ncols - 1) // ncols
    grid = np.zeros((nrows * 32, ncols * 32, 3))
    for i in range(n):
        row = i // ncols
        col = i % ncols
        grid[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = images[i]
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


# 训练循环
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, _) in enumerate(tqdm(trainloader, total=total_batches)):
        images = jt.array(images)

        # 随机时间步
        t = jt.randint(0, diffusion.T, (images.shape[0],)).float()
        xt, noise = diffusion.forward_process(images, t.int())

        # 预测噪声
        noise_pred = model(xt, t / diffusion.T)
        loss = nn.mse_loss(noise_pred, noise)

        optimizer.step(loss)

        epoch_loss += loss.item()

    avg_loss = epoch_loss / total_batches
    losses.append(avg_loss)
    logging.info(f'Epoch {epoch + 1}, Loss: {avg_loss:.6f}')
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.6f}')

    # 每5个 epoch 保存采样图像
    if (epoch + 1) % 5 == 0:
        model.eval()
        sample = jt.randn(16, 3, 32, 32)
        with jt.no_grad():  # 添加此行，禁用梯度
            for t in reversed(range(diffusion.T)):
                sample = diffusion.sample_step(model, sample, jt.array([t]))
                jt.gc() #主动垃圾回收
        save_image_grid(sample, f'samples/epoch_{epoch + 1}.png')

# 绘制 loss 曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('logs/loss_curve.png')
plt.close()

# 保存模型
jt.save(model.state_dict(), 'unet_diffusion_cifar10.jt')