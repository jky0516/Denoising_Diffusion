import torch
import numpy as np
from models.cDDPM_unet import cDDPM_UNet
from common.schedules import get_diffusion_schedule
from common.data_utils import load_lowKa, load_highKa_eval
from common.sampling import sample_all_in_batches
from common.plotting import scatter_plot, heatmap_pair
from samplers.cddpm import ddim_c_sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DDPM beta schedule
steps = 1000
_, _, alpha_bars = get_diffusion_schedule(steps, mode="linear", device=device)

# Load model weights
model = cDDPM_UNet().to(device)
model.load_state_dict(torch.load("Weight/cDDPM_Unet.pth", map_location=device))
model.eval()

# High Ka inference
high_loader, X_high_t, y_high_t = load_highKa_eval(
    high_files=("Data/X_scaled_zc_high.npy", "Data/target_scaled_zc_high.npy"),
    batch_size=128
)

recovered_high = sample_all_in_batches(
    sampler_fn=lambda model, **kw: ddim_c_sampler(model=model, alpha_bars=alpha_bars, ddim_steps=10, **kw),
    model=model,
    cond_input=X_high_t,
    batch_size=1000,
    device=device.type
)

# Plot results
scatter_plot(
    y_high_t.numpy().reshape(len(y_high_t), -1),
    recovered_high.numpy().reshape(len(recovered_high), -1),
    filename="highKa_scatter.png",
    title="High Ka"
)
heatmap_pair(y_high_t[205, 0].numpy(), recovered_high[205, 0].numpy(), "highKa_heatmap.png")

# Low Ka validation inference
_, _, X_train, y_train, X_val, y_val = load_lowKa(
    low_files=("Data/X_scaled_zc_low.npy", "Data/target_scaled_zc_low.npy"),
    batch_size=128,
    test_split=0.2,
    preprocess_fn=None,
    shuffle=False
)

recovered_low = sample_all_in_batches(
    sampler_fn=lambda model, **kw: ddim_c_sampler(model=model, alpha_bars=alpha_bars, ddim_steps=10, **kw),
    model=model,
    cond_input=X_val,
    batch_size=1000,
    device=device.type
)

# Plot results
scatter_plot(
    y_val.numpy().reshape(len(y_val), -1),
    recovered_low.numpy().reshape(len(recovered_low), -1),
    filename="lowKa_val_scatter.png",
    title="Low Ka Validation"
)
heatmap_pair(y_val[100, 0].numpy(), recovered_low[100, 0].numpy(), "lowKa_val_heatmap.png")
