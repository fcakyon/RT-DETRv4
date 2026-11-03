"""Convert the Hugging Face DINOv3 ViT-B/16 weights to the Meta layout that the RT-DETRv4 teacher loads via torch.hub.

Meta's `dinov3_vitb16(weights=...)` takes a plain state dict, while `facebook/dinov3-vitb16-pretrain-lvd1689m` ships the
transformers layout (separate q/k/v projections, `layer.N.*` names). The conversion renames the tensors, fuses q/k/v
into `attn.qkv` with a zero key bias (Meta masks it), keeps the deterministic RoPE periods and bias masks of a freshly
built model, loads the result strictly and checks that its patch tokens match the transformers model on random images
before writing OUT.

Usage, with HF_DIR the downloaded Hugging Face repo files (model.safetensors and config.json) and DINOV3 a
facebookresearch/dinov3 checkout:
    CUDA_VISIBLE_DEVICES= .venv/bin/python ul37/convert_dinov3.py HF_DIR DINOV3 OUT
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import DINOv3ViTModel

hf_dir, repo, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
hf = load_file(hf_dir / "model.safetensors")
model = torch.hub.load(repo, "dinov3_vitb16", source="local", pretrained=False)
state = model.state_dict()  # rope_embed.periods and the qkv bias masks are constants of the architecture

new = {
    "cls_token": hf["embeddings.cls_token"],
    "storage_tokens": hf["embeddings.register_tokens"],
    "mask_token": hf["embeddings.mask_token"].squeeze(0),
    "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
    "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
    "norm.weight": hf["norm.weight"],
    "norm.bias": hf["norm.bias"],
}
for i in range(sum(k.endswith(".norm1.weight") for k in hf)):
    h, m = f"layer.{i}.", f"blocks.{i}."
    attn = h + "attention."
    new |= {
        m + "attn.qkv.weight": torch.cat([hf[attn + f"{n}_proj.weight"] for n in "qkv"]),
        m + "attn.qkv.bias": torch.cat(
            [hf[attn + "q_proj.bias"], torch.zeros_like(hf[attn + "q_proj.bias"]), hf[attn + "v_proj.bias"]]
        ),
        m + "attn.proj.weight": hf[attn + "o_proj.weight"],
        m + "attn.proj.bias": hf[attn + "o_proj.bias"],
        m + "ls1.gamma": hf[h + "layer_scale1.lambda1"],
        m + "ls2.gamma": hf[h + "layer_scale2.lambda1"],
        m + "mlp.fc1.weight": hf[h + "mlp.up_proj.weight"],
        m + "mlp.fc1.bias": hf[h + "mlp.up_proj.bias"],
        m + "mlp.fc2.weight": hf[h + "mlp.down_proj.weight"],
        m + "mlp.fc2.bias": hf[h + "mlp.down_proj.bias"],
    }
    for n in ("norm1", "norm2"):
        new |= {m + f"{n}.weight": hf[h + f"{n}.weight"], m + f"{n}.bias": hf[h + f"{n}.bias"]}
model.load_state_dict(state | new, strict=True)
assert all(new[k].shape == state[k].shape for k in new), "a converted tensor changed shape"

reference = DINOv3ViTModel.from_pretrained(hf_dir).eval()
model.eval()
torch.manual_seed(0)
with torch.no_grad():
    for size in (224, 640):
        x = torch.randn(2, 3, size, size)
        ours = model.forward_features(x)["x_norm_patchtokens"]
        theirs = reference(pixel_values=x).last_hidden_state[:, 1 + reference.config.num_register_tokens :]
        diff = (ours - theirs).abs().max().item()
        print(f"{size}px max abs diff {diff:.2e}")
        assert diff < 1e-3, diff
torch.save(model.state_dict(), out)
print("wrote", out)
