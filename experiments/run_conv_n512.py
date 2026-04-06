"""N=512 2D training with conv trick — run this directly to see live output.

Usage:
    python -u experiments/run_conv_n512.py
"""
import torch
import numpy as np
import time
import json
from pathlib import Path
from src.data.poisson import assemble_poisson_2d
from src.models.unet import UNet
from src.training.losses import ConditionLoss
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 512
epochs = 300
steps = 100
probes = 32
probe_batch = 8  # small batch to fit VRAM
lr = 5e-4
base_features = 32

print(f'N={N} ({N*N:,} DOFs), base_features={base_features}, mode=conv')
print(f'Epochs: {epochs}, steps: {steps}, probes: {probes}, batch: {probe_batch}')
print(f'Device: {device}')

model = UNet(base_features=base_features, levels=3, dim=2).to(device)
params = sum(p.numel() for p in model.parameters())
print(f'Model: {params:,} params (ratio: {params/(N*N):.1f}x unknowns)')

# Load previous best if exists
best_path = Path('results/curriculum/2d/N512_bf32/best.pt')
if best_path.exists():
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    print(f'Loaded weights from {best_path}')
else:
    print('Training from scratch')

A = assemble_poisson_2d(N)
loss_fn = ConditionLoss(A, N, num_probes=probes, dim=2, mode='conv').to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = torch.amp.GradScaler('cuda')

save_dir = Path('results/curriculum/2d/N512_conv')
save_dir.mkdir(parents=True, exist_ok=True)

best_loss = float('inf')
loss_history = []
t_start = time.time()

# Benchmark one step first
print('\nBenchmarking one step...')
torch.cuda.synchronize()
t0 = time.perf_counter()
optimizer.zero_grad()
loss = loss_fn(model, device, use_amp=True, use_checkpointing=True, probe_batch_size=probe_batch)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
torch.cuda.synchronize()
step_time = time.perf_counter() - t0
print(f'One step: {step_time*1000:.0f} ms')
print(f'Estimated epoch: {step_time*steps:.0f} s = {step_time*steps/60:.1f} min')
print(f'Estimated total: {step_time*steps*epochs/3600:.1f} hours')
print(f'GPU VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB')
print()

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    epoch_start = time.time()

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model, device, use_amp=True, use_checkpointing=True, probe_batch_size=probe_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / steps
    loss_history.append(avg_loss)
    epoch_time = time.time() - epoch_start

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_dir / 'best.pt')
        marker = ' *'
    else:
        marker = ''

    elapsed = time.time() - t_start
    remaining = (epochs - epoch) * epoch_time
    print(f'Epoch {epoch:4d}/{epochs} | loss={avg_loss:.4f} | best={best_loss:.4f} | '
          f'epoch={epoch_time:.1f}s | elapsed={elapsed/60:.0f}m | remaining={remaining/60:.0f}m{marker}')

torch.save(model.state_dict(), save_dir / 'final.pt')
np.save(save_dir / 'loss_history.npy', np.array(loss_history))
config = {
    'N': N, 'dim': 2, 'epochs': epochs, 'steps': steps, 'lr': lr,
    'probes': probes, 'probe_batch': probe_batch, 'base_features': base_features,
    'params': params, 'best_loss': best_loss, 'mode': 'conv',
    'training_time_s': time.time() - t_start,
}
with open(save_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f'\nDone. Best loss: {best_loss:.4f}. Time: {(time.time()-t_start)/3600:.1f} hours')
print(f'Saved to {save_dir}')
