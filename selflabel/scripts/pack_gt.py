#!/usr/bin/env python
"""Pack the deep per-frame GT (one JSON per frame, buried under
<inst>/default/smplx/keypoints3d/) into a CLEAN shareable format:

  3DGroundTruth/SceneX/{user}_{action}_{L-O-R}.json

one self-describing file per instance holding all 30 frames. 30x fewer files,
human-browsable, easy to merge Scene1-5. Use unpack_gt.py to restore the
per-frame layout for training (preprocess.py).
"""
import os, json, glob, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)   # data/raw/SceneX (per-frame GT)
    ap.add_argument("--out", required=True)          # 3DGroundTruth/SceneX
    ap.add_argument("--scene_name", default=None)
    args = ap.parse_args()
    scene = args.scene_name or os.path.basename(os.path.normpath(args.scene_root))
    os.makedirs(args.out, exist_ok=True)
    kp_dirs = sorted(glob.glob(os.path.join(args.scene_root, "user*", "action*", "*-*-*",
                                            "default", "smplx", "keypoints3d")))
    n = 0
    for kp in kp_dirs:
        parts = kp.split(os.sep)
        user, action, inst = parts[-6], parts[-5], parts[-4]
        files = sorted(glob.glob(os.path.join(kp, "[0-9]*.json")))
        if not files:
            continue
        def rnd(fr):  # 4-dec xyz (0.1mm), 3-dec conf -> compact, plenty precise
            return [[round(j[0], 4), round(j[1], 4), round(j[2], 4), round(j[3], 3)] for j in fr]
        frames = [rnd(json.load(open(f))[0]["keypoints3d"]) for f in files]   # [T][25][4]
        layout = {"user1": "A", "user2": "A", "user3": "B", "user4": "B",
                  "user5": "C", "user6": "C"}.get(user, "single") if scene.lower() == "scene3" else "single"
        obj = {
            "scene": scene, "user": user, "action": action, "instance": inst, "layout": layout,
            "joints": "BODY_25",
            "frame_coords": "person-centric: x=lateral(L->R hip), y=forward, z=height-above-floor; meters",
            "fields": ["x", "y", "z", "conf"],
            "num_frames": len(frames),
            "keypoints3d": frames,                                       # [T][25][4]
        }
        udir = os.path.join(args.out, user); os.makedirs(udir, exist_ok=True)
        json.dump(obj, open(os.path.join(udir, f"{action}_{inst}.json"), "w"))
        n += 1
        if n % 500 == 0:
            print(f"  packed {n} ...", flush=True)
    print(f"DONE packed {n} instances -> {args.out}")


if __name__ == "__main__":
    main()
