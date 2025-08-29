import torch
import numpy as np
from models.DDPM_unet import DDPM_UNet
from models.Simple_regressor import SimpleRegressor
from common.schedules import get_diffusion_schedule
from common.data_utils import load_lowKa, load_highKa_eval
from common.sampling import sample_all_in_batches
from common.plotting import scatter_plot, heatmap_pair
from samplers.ddpm_regressor import ddim_regressor_guidance_sampler
from samplers.ddpm_regressor import ddpm_regressor_guidance_sampler
from samplers.ddpm import ddim_sampler
from samplers.ddpm import ddpm_sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DDPM beta schedule
steps = 1000
betas, alphas, alpha_bars = get_diffusion_schedule(steps, mode="linear", device=device)

# Load models
ddpm = DDPM_UNet().to(device)
ddpm.load_state_dict(torch.load("Weight/DDPM_Unet.pth", map_location=device))
ddpm.eval()

regressor = SimpleRegressor().to(device)
regressor.load_state_dict(torch.load("Weight/Simple_regressor_discrete.pth", map_location=device))
regressor.eval()
use_regressor = True   # can be False if not using guidance

def choose_sampler():
    if use_regressor:
        return lambda model, **kw: ddim_regressor_guidance_sampler( # can be ddpm_regressor_guidance_sampler
            model=ddpm,
            regressor=regressor,
            alpha_bars=alpha_bars,
            ddim_steps=10,
            guidance_strength=50,
            **kw
        )
    else:
        return lambda model, **kw: ddim_sampler( # can be ddpm_sampler
            model=ddpm,
            alpha_bars=alpha_bars,
            ddim_steps=10,
            **kw
        )

sampler = choose_sampler()

# Low Ka validation inference
_, _, _, _, X_val, y_val = load_lowKa(
    low_files=("Data/X_scaled_zc_low.npy", "Data/target_scaled_zc_low.npy"),
    batch_size=128,
    test_split=0.2,
    preprocess_fn=None,
    shuffle=False
)

recovered_low = sample_all_in_batches(
    sampler_fn=sampler,
    model=ddpm,
    cond_input=X_val,
    batch_size=500,
    device=device.type
)

scatter_plot(
    y_val.numpy().reshape(len(y_val), -1),
    recovered_low.numpy().reshape(len(recovered_low), -1),
    filename="lowKa_val_scatter.png",
    title="Low Ka Validation"
)
heatmap_pair(y_val[100, 0].numpy(), recovered_low[100, 0].numpy(), "lowKa_val_heatmap.png")

# High Ka inference
high_loader, X_high_t, y_high_t = load_highKa_eval(
    high_files=("Data/X_scaled_zc_high.npy", "Data/target_scaled_zc_high.npy"),
    batch_size=128
)

recovered_high = sample_all_in_batches(
    sampler_fn=sampler,
    model=ddpm,
    cond_input=X_high_t,
    batch_size=500,
    device=device.type
)

scatter_plot(
    y_high_t.numpy().reshape(len(y_high_t), -1),
    recovered_high.numpy().reshape(len(recovered_high), -1),
    filename="highKa_scatter.png",
    title="High Ka"
)
heatmap_pair(y_high_t[205, 0].numpy(), recovered_high[205, 0].numpy(), "highKa_heatmap.png")
