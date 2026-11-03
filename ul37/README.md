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
