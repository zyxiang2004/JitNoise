import math
import os
import jittor as jt
from jittor import nn, Module
from jittor.dataset import CIFAR10
from jittor import transform as transforms
from jittor.contrib import concat
import numpy as np
from tqdm import tqdm
from typing import Dict
from PIL import Image

jt.flags.use_cuda = 1  # 启用CUDA


# ================= 核心模块适配 =================
class CustomAdamW(nn.AdamW):
    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.current_lr = lr


class DownSample(Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def execute(self, x, temb, cemb):
        return self.c1(x) + self.c2(x)


class UpSample(Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, stride=2, padding=2, output_padding=1)

    def execute(self, x, temb, cemb):
        return self.c(self.t(x))


class AttnBlock(Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def execute(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).reshape(B, C, -1).transpose(0, 2, 1)
        k = self.proj_k(h).reshape(B, C, -1)
        v = self.proj_v(h).reshape(B, C, -1).transpose(0, 2, 1)

        attn = jt.bmm(q, k) * (C ** -0.5)
        attn = nn.softmax(attn, dim=-1)

        h = jt.bmm(attn, v).transpose(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class ResBlock(Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def execute(self, x, temb, labels):
        h = self.block1(x)
        h += self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1)
        h += self.cond_proj(labels).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attn(h)


class ConditionalEmbedding(Module):
    def __init__(self, num_labels, d_model, dim):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_labels + 1, d_model),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def execute(self, t):
        return self.condEmbedding(t)


class Swish(Module):
    def execute(self, x):
        return x * jt.sigmoid(x)


class TimeEmbedding(Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        emb = jt.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = jt.exp(-emb)
        pos = jt.arange(T).float()
        emb = pos.unsqueeze(1) * emb.unsqueeze(0)
        emb = jt.stack([jt.sin(emb), jt.cos(emb)], dim=-1).view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding(T, d_model),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.timembedding[0].weight.data = emb

    def execute(self, t):
        return self.timembedding(t)


class UNet(Module):
    def __init__(self, T, num_labels, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
        self.head = nn.Conv2d(3, ch, 3, padding=1)

        # Downsample
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, tdim, dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # Middle
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False)
        ])

        # Upsample
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(chs.pop() + now_ch, out_ch, tdim, dropout))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, padding=1)
        )

    def execute(self, x, t, labels):
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)

        for layer in self.middleblocks:
            h = layer(h, temb, cemb)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = concat([h, hs.pop()], dim=1)
                h = layer(h, temb, cemb)
            else:
                h = layer(h, temb, cemb)

        return self.tail(h)


# ================= 扩散过程适配 =================
class GaussianDiffusionTrainer(Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', jt.linspace(beta_1, beta_T, T))
        alphas = 1. - self.betas
        alphas_bar = jt.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', jt.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', jt.sqrt(1. - alphas_bar))

    def execute(self, x_0, labels):
        t = jt.randint(self.T, shape=(x_0.shape[0],))
        noise = jt.randn_like(x_0)
        x_t = self.sqrt_alphas_bar[t].reshape(-1, 1, 1, 1) * x_0 + \
              self.sqrt_one_minus_alphas_bar[t].reshape(-1, 1, 1, 1) * noise
        loss = nn.mse_loss(self.model(x_t, t, labels), noise)
        return loss.mean()


class GaussianDiffusionSampler(Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w

        self.register_buffer('betas', jt.linspace(beta_1, beta_T, T))
        alphas = 1. - self.betas
        alphas_bar = jt.cumprod(alphas, dim=0)
        alphas_bar_prev = jt.concat([jt.ones(1), alphas_bar[:-1]])
        self.register_buffer('coeff1', jt.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / jt.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def execute(self, x_T, labels):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = jt.full((x_T.shape[0],), time_step, dtype=jt.int)
            eps = self.model(x_t, t, labels)
            nonEps = self.model(x_t, t, jt.zeros_like(labels))
            eps = (1. + self.w) * eps - self.w * nonEps

            mean = self.coeff1[t].reshape(-1, 1, 1, 1) * x_t - \
                   self.coeff2[t].reshape(-1, 1, 1, 1) * eps
            var = self.posterior_var[t].reshape(-1, 1, 1, 1)

            if time_step > 0:
                noise = jt.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + jt.sqrt(var) * noise
        return x_t.clamp(-1, 1)


# ================= 训练和评估逻辑 =================
def train(modelConfig: Dict):
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    total_samples = len(dataset)
    batch_size = modelConfig["batch_size"]
    num_batches = total_samples // batch_size
    if total_samples % batch_size != 0:
        num_batches += 1  # 处理余数样本
        print(f"实际批次数: {num_batches} (总样本数={total_samples}, Batch Size={batch_size})")
    dataloader = dataset.set_attrs(
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # 模型初始化
    net_model = UNet(**{k: modelConfig[k] for k in [
        'T', 'num_labels', 'ch', 'ch_mult', 'num_res_blocks', 'dropout'
    ]})
    optimizer = CustomAdamW(net_model.parameters(),
                            lr=modelConfig["lr"],
                            weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])

    # 训练循环
    for epoch in range(modelConfig["epoch"]):
        net_model.train()
        pbar = tqdm(dataloader, total=num_batches, desc=f"Epoch {epoch}")  # 关键修改
        for batch_idx, (images, labels) in enumerate(pbar):
            optimizer.zero_grad()
            x_0 = images * 2 - 1  # 转换到[-1,1]范围
            labels = labels + 1
            # 10%概率使用空标签
            mask = jt.rand(labels.shape) < 0.1
            labels = jt.where(mask, jt.zeros_like(labels), labels)

            loss = trainer(x_0, labels)
            optimizer.step(loss)

            pbar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.current_lr
            })

        # 保存模型
        if epoch % 10 == 0:
            save_path = os.path.join(modelConfig["save_dir"], f"ckpt_{epoch}.pkl")
            net_model.save(save_path)


def save_image(tensor, path, nrow=10, padding=2, normalize=True):
    """
    将张量保存为图像网格
    参数:
        tensor (jt.Var): 图像张量，形状为 [B, C, H, W]，范围 [0, 1]
        path (str): 保存路径
        nrow (int): 每行显示的图像数量
        padding (int): 图像之间的像素间距
        normalize (bool): 是否将值范围归一化到 [0, 255]
    """
    # 将 Jittor 张量转换为 NumPy 数组
    tensor = tensor.numpy()

    # 检查张量维度并调整
    if tensor.ndim == 4:
        tensor = tensor.transpose(0, 2, 3, 1)  # BCHW -> BHWC
    elif tensor.ndim == 3:
        tensor = tensor.transpose(1, 2, 0)  # CHW -> HWC
    else:
        raise ValueError("Unsupported tensor shape")

    # 归一化到 [0, 255]
    if normalize:
        tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)

    # 计算图像网格布局
    batch_size, height, width, channel = tensor.shape
    ncol = min(nrow, batch_size)
    nrow = int(np.ceil(batch_size / ncol))

    # 创建空白画布
    grid_image = np.zeros(
        (nrow * height + (nrow - 1) * padding,
         ncol * width + (ncol - 1) * padding,
         channel),
        dtype=tensor.dtype
    )

    # 填充图像到网格
    for i in range(batch_size):
        row = i // ncol
        col = i % ncol
        y_start = row * (height + padding)
        x_start = col * (width + padding)
        grid_image[y_start:y_start + height, x_start:x_start + width] = tensor[i]

    # 保存为 PNG
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid_image.squeeze()).save(path)


def eval(modelConfig: Dict):
    net_model = UNet(**{k: modelConfig[k] for k in [
        'T', 'num_labels', 'ch', 'ch_mult', 'num_res_blocks', 'dropout'
    ]})
    net_model.load(modelConfig["test_load_weight"])
    net_model.eval()

    sampler = GaussianDiffusionSampler(net_model, modelConfig["beta_1"],
                                       modelConfig["beta_T"], modelConfig["T"], modelConfig["w"])

    # 生成样本（分批生成防止显存溢出）
    batch_size = 25  # 减小批量大小
    total_samples = 100
    samples_list = []

    with jt.no_grad():  # 禁用梯度计算
        for i in range(0, total_samples, batch_size):
            # 创建当前批次的噪声和标签
            current_batch_size = min(batch_size, total_samples - i)
            labels = jt.array([(i + k) // 10 for k in range(current_batch_size)]).long() + 1
            noisy = jt.randn((current_batch_size, 3, 32, 32))

            # 生成样本
            samples = sampler(noisy, labels)
            samples = (samples + 1) * 0.5  # 转换到[0,1]
            samples_list.append(samples)

    # 合并所有批次结果
    final_samples = jt.concat(samples_list, dim=0)
    save_image(final_samples, os.path.join(modelConfig["sampled_dir"], 'samples.png'), nrow=10)


# ================= 主函数 =================
if __name__ == '__main__':
    config = {
        "state": "eval",  # 训练train,测试eval
        "epoch": 100,
        "batch_size": 128,
        "T": 1000,
        "num_labels": 10,
        "ch": 128,
        "ch_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "lr": 2e-4,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "w": 1.5,
        "save_dir": "./",
        "sampled_dir": "./samples",
        "test_load_weight": "ckpt_90.pkl"
    }

    if config["state"] == "train":
        train(config)
    else:
        eval(config)