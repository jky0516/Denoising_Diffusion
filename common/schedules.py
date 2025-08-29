import torch
import math

# simple beta schedule for DDPM
def betas_for_alpha_bar(num_steps, alpha_bar_fn, max_beta=0.999, device="cuda"):
    betas = []
    for i in range(num_steps):
        t1, t2 = i / num_steps, (i + 1) / num_steps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, device=device)

def get_diffusion_schedule(steps=1000, mode="linear", device="cuda"):
    if mode == "linear":
        betas = torch.linspace(1e-4, 0.02, steps, device=device)
    elif mode == "cosine":
        betas = betas_for_alpha_bar(
            steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            device=device,
        )
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
