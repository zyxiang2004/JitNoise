import jittor as jt
from jittor import nn
import numpy as np

class DiffusionModel:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = jt.linspace(beta_start, beta_end, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = jt.cumprod(self.alpha, dim=0)
        self.t_losses = jt.zeros(T)
        self.t_counts = jt.zeros(T)

    def update_beta(self, epoch, total_epochs):
        if epoch % 5 == 0 and epoch > 0:
            avg_losses = self.t_losses / (self.t_counts + 1e-6)
            weights = jt.softmax(avg_losses / avg_losses.max())
            beta_new = weights * (self.beta[-1] - self.beta[0]) + self.beta[0]
            self.beta = beta_new.sort()[1]
            self.alpha = 1.0 - self.beta
            self.alpha_bar = jt.cumprod(self.alpha, dim=0)

    def forward_process(self, x0, t, noise=None):
        if noise is None:
            noise = jt.randn_like(x0)
        return jt.sqrt(self.alpha_bar[t]).reshape(-1,1,1,1)*x0 + jt.sqrt(1-self.alpha_bar[t]).reshape(-1,1,1,1)*noise, noise

    def sample_step(self, model, xt, t, y=None):
        model.eval()
        beta_t, alpha_t, alpha_bar_t = self.beta[t].reshape(-1,1,1,1), self.alpha[t].reshape(-1,1,1,1), self.alpha_bar[t].reshape(-1,1,1,1)
        noise_pred = model(xt, t/self.T, y)
        x_prev = (xt - beta_t / jt.sqrt(1 - alpha_bar_t) * noise_pred) / jt.sqrt(alpha_t)
        return x_prev + jt.sqrt(beta_t) * jt.randn_like(xt) if t > 0 else x_prev

    def progressive_sample(self, model, x, epoch, total_epochs, y=None):
        current_T = int(self.T * (1 - epoch / total_epochs * 0.5))
        steps = np.linspace(self.T-1, 0, current_T, dtype=int)
        for t in steps:
            x = self.sample_step(model, x, jt.array([t]), y)
        return x
