# UL37 benchmark runner

`run_ul37.py` fine-tunes RT-DETRv4 on each of the 37 UL37 detection datasets and saves the predictions the benchmark
scores. Per dataset it converts the benchmark's pinned YOLO dataset to COCO json, starts from the released COCO detector
without its class head, trains 100 epochs at global batch 16 (seed 0, AMP, 640 square) from the size's own config with
the LR scaled linearly to batch 16 and every epoch field rescaled to 100 epochs, then writes `predictions.json` (top 300
from the EMA of the best checkpoint), `final_ema.pth` and `args.yaml`. A dataset with `args.yaml` is skipped only when its
saved provenance, this repo's commit included, matches the current run.

The released configs distill from a frozen DINOv3 ViT-B/16 teacher, and the runner keeps that loss. The teacher loads
through `torch.hub` from a `facebookresearch/dinov3` checkout at `~/dev/dinov3` and the weights file
`rtv4/dinov3_vitb16_pretrain_lvd1689m.pth` under `--weights`.

## Setup

```bash
git clone --single-branch -b ul37 git@github.com:fcakyon/RT-DETRv4.git rtdetrv4 && cd rtdetrv4
git checkout SHA && bash ul37/setup_env.sh # uv sync --frozen
```

## Run

`--benchmark` is a UL37 benchmark checkout: runs read its `ul37_platform_uris.txt`, `ul37_datasets.txt` and
`.venv-ul37` runtime with `.config-ul37-081026`, and scoring runs its `val_ultrabench.py`. `--weights` holds
`rtv4/rtv4_hgnetv2_{s,m,l,x}_coco.pth` (the released checkpoints) and the teacher file above.

```bash
.venv/bin/python ul37/run_ul37.py rtv4 SIZE --project RUNS --benchmark BENCH --weights WEIGHTS --device 0
```

Results go to `RUNS/ultrabench-081026-rtv4-SIZE/multitrain`. An unsharded run scores itself into
`val_ultrabench.json`. For shards, give each process `--subset START:STOP` (or comma-separated dataset names) and the
same `--name`, then score once with `--val-only` after all shards finish.

## Latency export

`export_rtv4.py` writes the ONNX model and PyTorch FP32 reference outputs for the TensorRT parity check. Its docstring
lists the arguments and ONNX packages.

## COCO eval

`coco_eval.py` scores a released COCO detector on COCO val2017 and COCO-SAMA (labels below). Its docstring lists the
outputs. Our scores used faster-coco-eval 1.7.0 from a separate dir D
(`uv pip install --target D faster-coco-eval==1.7.0`).

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=D .venv/bin/python ul37/coco_eval.py configs/rtv4/rtv4_hgnetv2_SIZE_coco.yml \
  WEIGHTS/rtv4/rtv4_hgnetv2_SIZE_coco.pth --images coco/images/val2017 --name rtv4-SIZE --output OUT \
  --ann coco=coco/annotations/instances_val2017.json sama-coco=sama-coco/instances_val2017.json
```

## COCO-SAMA labels

COCO-SAMA re-annotates COCO val2017 images. Build its `instances_val2017.json` from Sama's release:

```bash
curl -LO https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-val.zip
echo "c55454f6657ce63891783136cc5dfe5f1e9c5e1748c0d8ca08b456a804eab15f  sama-coco-val.zip" | sha256sum -c
unzip -j sama-coco-val.zip '*sama_coco_coco_format_val_*.json'
python3 -c "import json; s = [json.load(open(f'sama_coco_coco_format_val_{i}.json')) for i in range(5)]; open('instances_val2017.json', 'w').write(json.dumps({k: s[0][k] for k in ('info', 'licenses', 'categories')} | {'images': sorted((x for d in s for x in d['images']), key=lambda x: x['id']), 'annotations': [x for d in s for x in d['annotations']]}))"
sha256sum instances_val2017.json # c8575a4b8eb39a8f7316c3e6f51349834ca0dc3b11e35f615497064c5d0b79b0
```
