import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

# Conditional sampling using DDPM with guidance
def ddpm_regressor_guidance_sampler(*, model, regressor, cond_input,
                                    betas, alphas, alpha_bars,
                                    guidance_strength=50, steps=1000, device="cuda"):
    num = cond_input.size(0)
    x = torch.randn(num, 1, 35, 31, device=device)
    for t_step in tqdm(reversed(range(steps)), desc="Sampling", ncols=80):
        t = torch.full((num,), t_step, device=device, dtype=torch.long).view(-1, 1)
        beta_t = betas[t].view(-1, 1, 1, 1)
        alpha_t = alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
        coef1 = 1 / alpha_t.sqrt()
        coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()

        x.requires_grad_(True)
        with torch.no_grad():
            pred_noise = model(x, t)

        pred_c = regressor(x, t)
        mse = F.mse_loss(pred_c, cond_input, reduction='sum')
        grad = torch.autograd.grad(-mse, x, retain_graph=False, create_graph=False)[0]
        guided_pred_noise = pred_noise - guidance_strength * grad
        x = x.detach()

        mean = coef1 * (x - coef2 * guided_pred_noise)
        if t_step > 0:
            x = mean + beta_t.sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x

# Conditional sampling using DDIM with guidance
def ddim_regressor_guidance_sampler(*, model, regressor, cond_input, alpha_bars,
                                    ddim_steps=10, steps=1000, guidance_strength=10, device="cuda", eta=0.0):
    num = cond_input.size(0)
    x = torch.randn(num, 1, 35, 31, device=device)
    timesteps = list(reversed(np.linspace(0, steps-1, num=ddim_steps, dtype=int)))
    for i, t_step in enumerate(timesteps):
        t = torch.full((num,), t_step, device=device, dtype=torch.long)
        a_t = alpha_bars[t].view(-1, 1, 1, 1)
        a_prev = torch.ones_like(a_t) if i == len(timesteps)-1 else alpha_bars[timesteps[i+1]].view(-1, 1, 1, 1)

        x.requires_grad_(True)
        with torch.no_grad():
            pred_noise = model(x, t.view(-1,1))

        pred_c = regressor(x, t)
        mse = F.mse_loss(pred_c, cond_input, reduction='sum')
        grad = torch.autograd.grad(-mse, x, retain_graph=False, create_graph=False)[0]
        grad = grad / (grad.norm() + 1e-8)

        pred_x0 = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
        sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
        dir_xt = torch.sqrt(1 - a_prev - sigma_t**2) * pred_noise
        mean_pred = a_prev.sqrt() * pred_x0 + dir_xt
        mean_pred = mean_pred - guidance_strength * grad

        if t_step > 0:
            x = mean_pred + sigma_t * torch.randn_like(x)
        else:
            x = mean_pred
        x = x.detach().requires_grad_(True)
    return x.detach()
