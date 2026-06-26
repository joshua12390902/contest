#!/usr/bin/env python
"""Captioning-oriented pose normalization for self-labeled GT.

Converts each instance's keypoints3d (camera1 metric frame, per-scene arbitrary
orientation) into a PERSON-CENTRIC, FLOOR-ALIGNED, FACING-ALIGNED frame so that
all scenes share one representation (no antennas / no cross-scene calibration
needed) while preserving everything captioning needs:

  z = height above floor   (keeps standing / lying-on-floor -> fall semantics)
  x = lateral  (L-hip -> R-hip, floor-projected)   (keeps left/right semantics)
  y = forward  (= up x lateral)                     (facing-aligned, per instance)
  origin (x,y) = pelvis (MidHip) per frame          (drops room position)

Idempotent via a .normalized marker. Run ONCE on raw labeled instances.
BODY_25 indices: Nose0 Neck1 RSho2 RElb3 RWri4 LSho5 LElb6 LWri7 MidHip8
RHip9 RKnee10 RAnk11 LHip12 LKnee13 LAnk14 ... feet 19-24.
"""
import os, sys, json, glob, argparse
import numpy as np

FEET = [19, 20, 21, 22, 23, 24, 11, 14]   # toes/heels/ankles -> floor plane
MIDHIP, RHIP, LHIP, NOSE = 8, 9, 12, 0


def load_instance(kp_dir):
    files = sorted(glob.glob(os.path.join(kp_dir, "[0-9]*.json")))
    P, C = [], []
    for f in files:
        rows = json.load(open(f))[0]["keypoints3d"]
        a = np.array(rows, float)
        P.append(a[:, :3]); C.append(a[:, 3])
    return files, np.array(P), np.array(C)   # [T,25,3], [T,25]


def normalize(P, C):
    valid = C > 0
    pts = P[valid]
    # floor plane from feet joints (across all frames)
    feet = P[:, FEET, :].reshape(-1, 3); fc = C[:, FEET].reshape(-1)
    feet = feet[fc > 0]
    c = feet.mean(0); _, _, Vt = np.linalg.svd(feet - c); up = Vt[-1]
    head = np.nanmedian(P[:, NOSE], 0)
    if up @ (head - c) < 0:
        up = -up                                   # up points feet -> head
    # facing: lateral = L-hip -> R-hip, floor-projected, median over frames
    lat = P[:, RHIP] - P[:, LHIP]
    lat = lat[(C[:, RHIP] > 0) & (C[:, LHIP] > 0)]
    lat = np.median(lat, 0); lat = lat - up * (up @ lat); lat /= np.linalg.norm(lat)
    fwd = np.cross(up, lat)                         # right-handed: x=lat, y=fwd, z=up
    out = np.zeros_like(P)
    for t in range(P.shape[0]):
        pelvis = P[t, MIDHIP]
        d = P[t] - pelvis                           # relative to pelvis
        out[t, :, 0] = d @ lat                      # x lateral
        out[t, :, 1] = d @ fwd                      # y forward
        out[t, :, 2] = (P[t] - c) @ up              # z = height above floor (absolute)
    out[~valid] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", required=True)   # .../data/raw/SceneX
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    inst_dirs = sorted(glob.glob(os.path.join(args.scene_root, "user*", "action*", "*-*-*",
                                              "default", "smplx", "keypoints3d")))
    done = norm = skip = 0
    for kp_dir in inst_dirs:
        marker = os.path.join(kp_dir, ".normalized")
        if os.path.exists(marker) and not args.force:
            skip += 1; continue
        files, P, C = load_instance(kp_dir)
        if len(files) == 0:
            continue
        try:
            Pn = normalize(P, C)
        except Exception as e:
            print(f"[skip] {kp_dir}: {type(e).__name__}: {str(e)[:60]}"); continue
        for f, p, c in zip(files, Pn, C):
            rows = [[float(p[j, 0]), float(p[j, 1]), float(p[j, 2]), float(c[j])] for j in range(p.shape[0])]
            json.dump([{"keypoints3d": rows}], open(f, "w"))
        open(marker, "w").write("person-centric floor-aligned facing-aligned")
        norm += 1
        if norm % 200 == 0:
            print(f"  normalized {norm} ...", flush=True)
    print(f"DONE normalized={norm} skip(already)={skip} of {len(inst_dirs)} instances")


if __name__ == "__main__":
    main()
