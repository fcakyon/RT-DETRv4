"""Export RT-DETRv4 COCO detectors to static batch-1 FP32 ONNX at 640 and save their PyTorch FP32 references.

The graph is the repo's deploy model alone, one `images` input to the raw `pred_logits` / `pred_boxes` heads as for
RF-DETR and GTR, so the top-300 postprocessor that the repo's export_onnx.py appends is not timed. Weights load strictly
and `num_classes` comes from the checkpoint. References run on CPU over seed-0 noise and the given images, keyed by file
stem (Ultralytics' bus.jpg, which is also the export input, and zidane.jpg, COCO val2017 000000000139.jpg and
000000000285.jpg), normalized as the config's val transforms do. Writes OUT/rtv4-<size>.onnx and .ref.npz and merges
the size into OUT/export_info.json. One size per process, since the repo's config registry carries options such as
`use_lab` from one built model into the next. The repo is the fcakyon fork checkout holding this file, whose HEAD lands
in export_info.json as `repo`. WEIGHTS is the released weights root holding rtv4/, as for run_ul37.py.

Usage, with D an `uv pip install --target D --no-deps` dir holding onnx==1.23.2 ml-dtypes==0.6.0 onnxsim==0.7.3:
    CUDA_VISIBLE_DEVICES= PYTHONPATH=D .venv/bin/python ul37/export_rtv4.py OUT SIZE WEIGHTS IMAGE...
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn.functional as F
import torchvision

out, s = Path(sys.argv[1]), sys.argv[2]
REPO, WEIGHTS = Path(__file__).resolve().parents[1], Path(sys.argv[3]) / "rtv4"
images = [Path(p) for p in sys.argv[4:]]
OPSET = 16  # with onnxsim, as the repo's export_onnx.py defaults
sys.path.insert(0, str(REPO))
from engine.core import YAMLConfig


class Raw(torch.nn.Module):
    """Return the deploy model's raw heads as a (logits, boxes) tuple."""

    def __init__(self, model):
        """Wrap an RT-DETRv4 deploy model."""
        super().__init__()
        self.model = model

    def forward(self, images):
        """Run the model on normalized images."""
        y = self.model(images)
        return y["pred_logits"], y["pred_boxes"]


config, weights = f"configs/rtv4/rtv4_hgnetv2_{s}_coco.yml", f"rtv4_hgnetv2_{s}_coco.pth"
state = torch.load(WEIGHTS / weights, map_location="cpu", weights_only=False)["ema"]["module"]
nc = state["decoder.enc_score_head.bias"].numel()
cfg = YAMLConfig(str(REPO / config), num_classes=nc)
cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
cfg.model.load_state_dict(state)
model = Raw(cfg.model.deploy()).eval()

ops = cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
norm = next((o for o in ops if o["type"] == "Normalize"), {"mean": [0.0] * 3, "std": [1.0] * 3})
mean, std = (torch.tensor(norm[k]).view(1, 3, 1, 1) for k in ("mean", "std"))
torch.manual_seed(0)
xs = {"randn": torch.randn(1, 3, 640, 640)}
for p in images:
    img = torchvision.io.read_image(str(p), torchvision.io.ImageReadMode.RGB)[None].float() / 255
    xs[p.stem] = (F.interpolate(img, (640, 640), mode="bilinear", align_corners=False) - mean) / std

name = f"rtv4-{s}"
arrays = {}
with torch.no_grad():
    for key, x in xs.items():
        logits, boxes = model(x)
        arrays |= {f"{key}/images": x, f"{key}/pred_logits": logits, f"{key}/pred_boxes": boxes}
        top = logits.sigmoid().flatten(1).topk(5)
        print(name, key, [(round(v, 3), c % nc) for v, c in zip(top.values[0].tolist(), top.indices[0].tolist())])
np.savez(out / f"{name}.ref.npz", **{k: v.float().numpy() for k, v in arrays.items()})

path = out / f"{name}.onnx"
torch.onnx.export(
    model,
    (xs["bus"],),
    str(path),
    input_names=["images"],
    output_names=["pred_logits", "pred_boxes"],
    opset_version=OPSET,
    do_constant_folding=True,
    dynamo=False,
)
simplified, ok = onnxsim.simplify(str(path))
assert ok, f"{name} onnxsim check failed"
onnx.save(simplified, path)
out_json = out / "export_info.json"
info = json.loads(out_json.read_text()) if out_json.exists() else {}
repo = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True)
info[s] = {
    "config": config,
    "weights": str(WEIGHTS / weights),
    "params": sum(p.numel() for p in model.parameters()),
    "nc": nc,
    "repo": f"rtv4 {repo.strip()}",
    "opset": OPSET,
    "onnxsim": True,
    "normalize": norm["mean"] != [0.0] * 3,
}
print("EXPORTED", name, json.dumps(info[s]), flush=True)
out_json.write_text(json.dumps(info, indent=2))
print("rtv4", "torch", torch.__version__, "onnx", onnx.__version__)
