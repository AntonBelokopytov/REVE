"""Smoke test: load REVE-Base + position bank and run one forward pass."""
import os
os.environ["HF_HOME"] = "/media/alex/DATA1/REVE/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import torch
from transformers import AutoModel

print("torch", torch.__version__, "cuda", torch.cuda.is_available())

print("\n[1/3] Loading reve-positions ...")
pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
print("  class:", type(pos_bank).__name__)

print("\n[2/3] Loading reve-base ...")
model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)
model.eval()
print("  class:", type(model).__name__)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params: {n_params:,}")

cfg = getattr(model, "config", None)
if cfg is not None:
    print("  config keys:")
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            print(f"    {k} = {v}")

print("\n[3/3] Forward pass with 10-channel dummy tensor (4 s @ 200 Hz) ...")
electrode_names = ["Fz", "Cz", "Pz", "Oz", "F3", "F4", "C3", "C4", "P3", "P4"]
eeg = torch.randn(1, len(electrode_names), 800)

positions = pos_bank(electrode_names)
print("  positions shape:", tuple(positions.shape))
positions = positions.expand(eeg.size(0), -1, -1)

with torch.no_grad():
    out = model(eeg, positions)

if isinstance(out, torch.Tensor):
    print("  output (tensor):", tuple(out.shape))
else:
    print("  output type:", type(out).__name__)
    for name in ("last_hidden_state", "hidden_states", "pooler_output", "latent"):
        v = getattr(out, name, None)
        if isinstance(v, torch.Tensor):
            print(f"    {name}: {tuple(v.shape)}")
    if hasattr(out, "keys"):
        print("    keys:", list(out.keys()))
