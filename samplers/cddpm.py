import numpy as np
import torch
from tqdm.auto import tqdm

# Conditional sampling using DDIM
def ddim_c_sampler(*, model, cond_input, alpha_bars, steps=1000, ddim_steps=10, eta=0.0, device="cuda"):
    num = cond_input.size(0)
    x = torch.randn(num, 1, 35, 31, device=device)
    timesteps = list(reversed(np.linspace(0, steps-1, num=ddim_steps, dtype=int)))
    with torch.no_grad():
        for i, t_step in enumerate(timesteps):
            t = torch.full((num,), t_step, device=device, dtype=torch.long).view(-1, 1)
            a_t = alpha_bars[t].view(-1, 1, 1, 1)
            a_prev = torch.ones_like(a_t) if i == len(timesteps)-1 else alpha_bars[timesteps[i+1]].view(-1, 1, 1, 1)
            pred_noise = model(x, t, cond_input)
            pred_x0 = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
            dir_xt = torch.sqrt(1 - a_prev - sigma_t**2) * pred_noise
            mean_pred = a_prev.sqrt() * pred_x0 + dir_xt
            x = mean_pred if t_step == 0 else mean_pred + sigma_t * torch.randn_like(x)
    return x
