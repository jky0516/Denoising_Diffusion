import torch
import torch.nn as nn
from models.cDDPM_unet import cDDPM_UNet
from common.schedules import get_diffusion_schedule
from common.data_utils import load_lowKa
from common.preprocessing import log_transform
from common.training_utils import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, *_ = load_lowKa(
    low_files=("Data/X_scaled_zc_low.npy", "Data/target_scaled_zc_low.npy"),
    batch_size=128, test_split=0.2, preprocess_fn=log_transform, shuffle=True
)

# Establish model, optimizer and loss function
model = cDDPM_UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# DDPM schedulw
steps = 1000
_, _, alpha_bars = get_diffusion_schedule(steps, mode="linear", device=device)

# Training loop
train_model(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    steps=steps,
    alpha_bars=alpha_bars,
    loss_fn=loss_fn,
    epochs=800,
    save_path="cDDPM_Unet.pth",
    patience=100,
    device=device.type
)
