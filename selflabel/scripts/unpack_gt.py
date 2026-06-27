#!/usr/bin/env python
"""Restore the clean 3DGroundTruth/SceneX/{inst}.json files back into the
per-frame layout that preprocess.py expects (next to the CSI):

  data/raw/SceneX/<user>/<action>/<L-O-R>/default/smplx/keypoints3d/000000.json ...

Run this before preprocess when training from the shared GT format.
"""
import os, json, glob, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)       # 3DGroundTruth/SceneX
    ap.add_argument("--scene_root", required=True)   # data/raw/SceneX (where CSI lives)
    args = ap.parse_args()
    n = 0
    files = glob.glob(os.path.join(args.gt_dir, "*.json")) + glob.glob(os.path.join(args.gt_dir, "*", "*.json"))
    for f in sorted(files):
        obj = json.load(open(f))
        inst_dir = os.path.join(args.scene_root, obj["user"], obj["action"], obj["instance"])
        kp = os.path.join(inst_dir, "default", "smplx", "keypoints3d")
        os.makedirs(kp, exist_ok=True)
        for t, frame in enumerate(obj["keypoints3d"]):
            json.dump([{"keypoints3d": frame}], open(os.path.join(kp, f"{t:06d}.json"), "w"))
        n += 1
        if n % 500 == 0:
            print(f"  unpacked {n} ...", flush=True)
    print(f"DONE unpacked {n} instances -> {args.scene_root}")


if __name__ == "__main__":
    main()
