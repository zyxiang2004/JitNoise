import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import jittor as jt
from jittor import nn, optim, dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import jittor.transform as transforms

from models.unet import UNet, FeatureExtractor
from models.ddpm import DiffusionModel

jt.flags.use_cuda = 1

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = dataset.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = dataset.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

# Model & Setup
model = UNet(num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=2e-4)
diffusion = DiffusionModel(T=1000)
feature_extractor = FeatureExtractor()
feature_extractor.eval()

# Directories
os.makedirs('samples', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def perceptual_loss(pred, target):
    pred = (pred + 1) / 2
    target = (target + 1) / 2
    mean = jt.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = jt.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    pred = (pred - mean) / std
    target = (target - mean) / std
    f_pred = feature_extractor(pred)
    f_target = feature_extractor(target)
    return sum(nn.mse_loss(fp, ft) for fp, ft in zip(f_pred, f_target))

def save_image_grid(images, filename, nrow=4):
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.numpy().transpose(0, 2, 3, 1)
    grid = np.zeros((nrow*32, nrow*32, 3))
    for i in range(images.shape[0]):
        grid[(i//nrow)*32:(i//nrow+1)*32, (i%nrow)*32:(i%nrow+1)*32] = images[i]
    plt.imsave(filename, grid)

# Logger
logging.basicConfig(filename='logs/training.log', level=logging.INFO)
losses = []
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm(trainloader):
        images = jt.array(images)
        labels = jt.array(labels)
        t = jt.randint(0, diffusion.T, (images.shape[0],)).float()
        xt, noise = diffusion.forward_process(images, t.int())
        noise_pred = model(xt, t / diffusion.T, labels)
        mse = nn.mse_loss(noise_pred, noise)
        perc = perceptual_loss(noise_pred, noise)
        loss = mse + 0.1 * perc
        optimizer.step(loss)
        epoch_loss += loss.item()
        for ti in t.int().numpy():
            diffusion.t_losses[ti] += loss.item()
            diffusion.t_counts[ti] += 1
    avg_loss = epoch_loss / len(trainloader)
    losses.append(avg_loss)
    logging.info(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')
    diffusion.update_beta(epoch, num_epochs)
    if (epoch+1) % 5 == 0:
        model.eval()
        sample = jt.randn(16, 3, 32, 32)
        sample_labels = jt.array(np.random.randint(0, 10, 16))
        sample = diffusion.progressive_sample(model, sample, epoch, num_epochs, sample_labels)
        save_image_grid(sample, f'samples/epoch_{epoch+1}.png')

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("logs/loss_curve.png")

jt.save(model.state_dict(), 'checkpoints/unet_diffusion_cifar10.jt')
