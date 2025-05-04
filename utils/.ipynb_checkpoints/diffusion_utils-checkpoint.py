import jittor as jt
import math

def make_beta_schedule(schedule, timesteps, start=1e-4, end=0.02):
    if schedule == "linear":
        return jt.linspace(start, end, timesteps)
    elif schedule == "cosine":
        steps = timesteps + 1
        x = jt.linspace(0, timesteps, steps) / timesteps
        alphas = jt.cos((x + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return jt.clamp(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule}")

def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jt.exp(jt.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jt.concat([jt.sin(emb), jt.cos(emb)], dim=1)
    return emb
