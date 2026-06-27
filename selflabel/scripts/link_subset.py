#!/usr/bin/env python
"""Build a symlink subset dir from an explicit instance list, so preprocess.py
materializes .pt only for the chosen (CSI+GT complete) instances.

  python selflabel/scripts/link_subset.py --scene_root PerceptAlign/data/raw/Scene1 \
      --instances_file subset_scene1.txt --out PerceptAlign/data/train_subset/Scene1
"""
import os, glob, argparse


def has_data(d):
    csi = glob.glob(os.path.join(d, "csi_mat", "*.mat"))
    gt = os.path.exists(os.path.join(d, "default", "smplx", "keypoints3d", "000029.json"))
    return len(csi) >= 3 and gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)
    ap.add_argument("--instances_file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.scene_root); out = os.path.abspath(args.out)
    insts = [l.strip() for l in open(args.instances_file) if l.strip()]
    linked = missing = 0
    for inst in insts:
        src = os.path.join(root, inst)
        if not has_data(src):
            missing += 1; continue
        link = os.path.join(out, inst)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(src, link)
        linked += 1
    print(f"{args.out}: linked={linked} missing(no csi/gt)={missing} of {len(insts)}")


if __name__ == "__main__":
    main()
