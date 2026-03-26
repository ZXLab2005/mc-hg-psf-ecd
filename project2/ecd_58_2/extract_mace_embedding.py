#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_mace_embedding.py
----------------------------------
MACE-MP 提取 embedding（适配 mace-torch 0.3.14）
python extract_mace_embedding.py \
  --xyz_dir ./xyz \
  --out_csv mace_emb.csv \
  --model ./mace_medium.model \
  --device cpu
? --model 支持 small/medium/large 或本地 .model
? 本地模型优先
? 自动重试下载/加载
? num_layers=None 时自动读取模型层数（否则 mace 0.3.x 会报错）

输出：
id, emb_0, emb_1, ..., emb_{d-1}
"""

import os, glob, argparse, time
import numpy as np
import pandas as pd
from ase.io import read


def load_mace_mp(model_spec, device="cpu", retries=3, wait_s=5):
    from mace.calculators import mace_mp
    last_err = None
    for i in range(retries):
        try:
            print(f"[MACE] Loading MACE-MP ({model_spec}) on {device} ... try {i+1}/{retries}")
            calc = mace_mp(model=model_spec, device=device, default_dtype="float32")
            return calc
        except Exception as e:
            last_err = e
            print(f"[MACE] load failed: {e}")
            if i < retries - 1:
                print(f"[MACE] retry after {wait_s}s ...")
                time.sleep(wait_s)
    raise RuntimeError(f"Model load failed after {retries} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz_dir", default="./xyz")
    ap.add_argument("--out_csv", default="mace_emb.csv")
    ap.add_argument("--model", default="medium",
                    help='("small/medium/large") or local .model path')
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--num_layers", type=int, default=None,
                    help="取前 n 层 descriptors；默认自动读取模型层数")
    ap.add_argument("--full_desc", action="store_true",
                    help="若开启则 invariants_only=False（一般不需要）")
    args = ap.parse_args()

    # ---- model 参数：支持本地路径 ----
    model_spec = args.model
    if os.path.exists(model_spec):
        model_spec = os.path.abspath(model_spec)
        print(f"[MACE] Using local checkpoint: {model_spec}")
    else:
        if model_spec not in ["small", "medium", "large"]:
            raise ValueError(f"--model must be small/medium/large or local path, got {args.model}")
        print(f"[MACE] Using foundation model: {model_spec}")

    # ---- 加载 calculator ----
    calc = load_mace_mp(model_spec, device=args.device, retries=3, wait_s=6)

    # ---- 自动确定 num_layers（兼容 mace 0.3.x）----
    if args.num_layers is None:
        nL = None
        m0 = calc.models[0]
        for key in ["num_interactions", "num_layers", "n_layers"]:
            if hasattr(m0, key):
                nL = int(getattr(m0, key))
                break
        if nL is None:
            for key in ["interaction_blocks", "interactions"]:
                if hasattr(m0, key):
                    try:
                        nL = len(getattr(m0, key))
                        break
                    except Exception:
                        pass
        if nL is None:
            nL = 2
        args.num_layers = nL
        print(f"[MACE] auto num_layers = {args.num_layers}")

    # ---- 读取 xyz ----
    xyz_files = sorted(glob.glob(os.path.join(args.xyz_dir, "*.xyz")))
    if not xyz_files:
        raise FileNotFoundError(f"No xyz files under {args.xyz_dir}")

    rows = []
    emb_dim = None
    print(f"[MACE] Found {len(xyz_files)} xyz files.")

    for i, f in enumerate(xyz_files, start=1):
        sid = os.path.splitext(os.path.basename(f))[0]
        atoms = read(f)

        # 0.3.x 稳定接口：get_descriptors 必须给 num_layers=int
        desc = calc.get_descriptors(
            atoms,
            invariants_only=(not args.full_desc),
            num_layers=args.num_layers
        )  # (n_atoms, d)

        emb = desc.mean(axis=0).astype(np.float32)

        if emb_dim is None:
            emb_dim = emb.shape[0]
            print(f"[MACE] embedding dim = {emb_dim}")

        rows.append([sid] + emb.tolist())

        if i % 5 == 0 or i == len(xyz_files):
            print(f"[MACE] {i}/{len(xyz_files)} done")

    cols = ["id"] + [f"emb_{k}" for k in range(emb_dim)]
    pd.DataFrame(rows, columns=cols).to_csv(args.out_csv, index=False)
    print(f"[DONE] Saved MACE embeddings -> {args.out_csv}")


if __name__ == "__main__":
    main()
