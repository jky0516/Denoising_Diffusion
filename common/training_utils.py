import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Random noise generation and noise adding
def add_noise(img, t, alpha_bars):
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(img)
    noisy = alpha_bar_t.sqrt() * img + (1 - alpha_bar_t).sqrt() * noise
    return noisy, noise

# Training loop
def train_model(model, optimizer, train_loader, val_loader, steps, alpha_bars,
                loss_fn, epochs, save_path, patience=50, device="cuda"):
    best_val_loss = float('inf')
    counter = 0
    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for cond_input, img in tqdm(train_loader, desc=f"Epoch {epoch}"):
            cond_input, img = cond_input.to(device), img.to(device)
            t = torch.randint(0, steps, (cond_input.size(0),), device=device).view(-1, 1)
            noisy_img, noise = add_noise(img, t, alpha_bars)
            pred = model(noisy_img, t) if model.__class__.__name__ != "cDDPM_UNet" else model(noisy_img, t, cond_input)
            loss = loss_fn(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for cond_input, img in val_loader:
                    cond_input, img = cond_input.to(device), img.to(device)
                    t = torch.randint(0, steps, (cond_input.size(0),), device=device).view(-1, 1)
                    noisy_img, noise = add_noise(img, t, alpha_bars)
                    pred = model(noisy_img, t) if model.__class__.__name__ != "cDDPM_UNet" else model(noisy_img, t, cond_input)
                    val_loss += loss_fn(pred, noise).item()

        train_loss /= max(1, len(train_loader))
        val_loss = val_loss / max(1, len(val_loader)) if val_loader is not None else float('nan')
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f"[Epoch {epoch}] Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
            print(f"Saved model with val_loss={val_loss:.4f}")
        elif val_loader is not None:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), save_path)

    # Plot loss curves
    plt.figure(figsize=(4, 3))
    plt.plot(train_loss_history, label='train_loss')
    if val_loader is not None:
        plt.plot(val_loss_history, label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Curve')
    plt.legend(); plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path.replace(".pth", "_training_curve.png"))
