import torch
from tqdm.auto import tqdm
from .preprocessing import inverse_bct, normalize_nonnegative

# Sampling the results batch by batch to save memory
def sample_all_in_batches(sampler_fn, model, cond_input, batch_size=64, device="cuda", postprocess=True, **kwargs):
    total = cond_input.size(0)
    all_samples = []
    for i in tqdm(range(0, total, batch_size), desc="Sampling batches"):
        end = min(i + batch_size, total)
        batch = cond_input[i:end].to(device)
        samples = sampler_fn(model=model, cond_input=batch, device=device, **kwargs)
        if postprocess:
            samples = inverse_bct(samples, lamb=0.1)
            samples = normalize_nonnegative(samples)
        all_samples.append(samples.detach().cpu())
    return torch.cat(all_samples, dim=0)
