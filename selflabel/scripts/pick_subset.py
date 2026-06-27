#!/usr/bin/env python
"""Pick a balanced per-user subset of instances from a packed 3DGroundTruth/SceneX
dir, emit an instance list (user/action/L-O-R per line) for download/preprocess.

  python selflabel/scripts/pick_subset.py --gt_dir 3DGroundTruth/Scene1 --per_user 80 --out subset_scene1.txt
"""
import os, glob, json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--per_user", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    users = sorted([d for d in os.listdir(args.gt_dir)
                    if d.startswith("user") and os.path.isdir(os.path.join(args.gt_dir, d))],
                   key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)
    lines = []
    for u in users:
        files = sorted(glob.glob(os.path.join(args.gt_dir, u, "*.json")))[: args.per_user]
        for f in files:
            stem = os.path.basename(f)[:-5]           # action_L-O-R
            action, inst = stem.split("_", 1)
            lines.append(f"{u}/{action}/{inst}")
        print(f"  {u}: picked {len(files)}")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"-> {args.out}  total={len(lines)}")


if __name__ == "__main__":
    main()
