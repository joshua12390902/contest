#!/usr/bin/env python
"""Build a balanced, GT-complete subset of a raw scene as a symlink tree, so that
preprocess.py only materializes .pt for a bounded number of instances (each .pt is
~52MB; preprocessing all thousands would overflow disk).

Picks up to --per_user instances PER user that already have full GT
(default/smplx/keypoints3d/000029.json present), and symlinks them under --out.

  python selflabel/scripts/make_subset.py \
      --scene_root PerceptAlign/data/raw/Scene4 \
      --out PerceptAlign/data/train_subset/Scene4 --per_user 120
"""
import argparse, glob, os


def has_gt(inst_dir):
    return os.path.exists(os.path.join(inst_dir, "default", "smplx", "keypoints3d", "000029.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_user", type=int, default=120)
    args = ap.parse_args()
    scene_root = os.path.abspath(args.scene_root)
    out = os.path.abspath(args.out)

    users = sorted(d for d in os.listdir(scene_root)
                   if d.startswith("user") and os.path.isdir(os.path.join(scene_root, d)))
    total = 0
    for u in users:
        insts = sorted(glob.glob(os.path.join(scene_root, u, "action*", "*-*-*")))
        insts = [d for d in insts if has_gt(d)]
        picked = insts[: args.per_user]
        for d in picked:
            rel = os.path.relpath(d, scene_root)
            link = os.path.join(out, rel)
            os.makedirs(os.path.dirname(link), exist_ok=True)
            if not os.path.lexists(link):
                os.symlink(d, link)
        total += len(picked)
        print(f"  {u}: gt={len(insts)} picked={len(picked)}")
    print(f"subset -> {out}  total={total}")


if __name__ == "__main__":
    main()
