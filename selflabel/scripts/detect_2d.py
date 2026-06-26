#!/usr/bin/env python
"""Step A: 2D keypoint detection on the 3 camera views of one instance.

Single-person empty-room scene -> take the largest-bbox person per frame.
Subsample every 5th video frame so we get 30 GT frames (matches the repo's
30-keypoints-per-instance convention; feeding 150 would corrupt CSI alignment).

Outputs (per instance):
  <out>/kpts2d.npz   : views x [30,133,2] xy + [30,133] score  (COCO-WholeBody 133)
  <out>/debug/<view>_overlay.png : sanity overlays
"""
import os, sys, argparse
import numpy as np
import cv2
from rtmlib import Wholebody, draw_skeleton

VIEWS = ["output1", "output2", "output3"]
SUBSAMPLE = 5          # 150 frames @30fps -> 30 frames @6fps
N_GT = 30


def read_subsampled(video_path):
    cap = cv2.VideoCapture(video_path)
    frames, idx, kept = [], 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % SUBSAMPLE == 0:
            frames.append(fr); kept.append(idx)
        idx += 1
    cap.release()
    return frames, kept


def largest_person(kpts, scores):
    if kpts.shape[0] == 0:
        return None, None
    areas = [(k[:, 0].max() - k[:, 0].min()) * (k[:, 1].max() - k[:, 1].min()) for k in kpts]
    i = int(np.argmax(areas))
    return kpts[i], scores[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "debug"), exist_ok=True)

    wb = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")  # CPU: ~0.16s/frame, plenty

    all_xy, all_sc, summary = {}, {}, []
    for v in VIEWS:
        vp = os.path.join(args.instance, v, "segment_1.mp4")
        frames, kept = read_subsampled(vp)
        frames = frames[:N_GT]
        xy = np.full((N_GT, 133, 2), np.nan, np.float32)
        sc = np.zeros((N_GT, 133), np.float32)
        miss = 0
        for t, fr in enumerate(frames):
            k, s = wb(fr)
            pk, ps = largest_person(k, s)
            if pk is None:
                miss += 1; continue
            xy[t], sc[t] = pk, ps
        all_xy[v], all_sc[v] = xy, sc
        body_conf = sc[:, :23]                      # body(17)+feet(6) = the joints we triangulate
        summary.append((v, len(frames), miss, float(body_conf.mean()),
                        float((body_conf > 0.5).mean())))
        # overlay middle frame
        mid = min(15, len(frames) - 1)
        ov = draw_skeleton(frames[mid].copy(), all_xy[v][mid:mid+1], all_sc[v][mid:mid+1], kpt_thr=0.3)
        cv2.imwrite(os.path.join(args.out, "debug", f"{v}_overlay.png"), ov)

    np.savez_compressed(os.path.join(args.out, "kpts2d.npz"),
                        **{f"{v}_xy": all_xy[v] for v in VIEWS},
                        **{f"{v}_sc": all_sc[v] for v in VIEWS})
    print(f"{'view':9} {'frames':>6} {'miss':>4} {'bodyConf':>9} {'frac>0.5':>9}")
    for v, n, m, mc, fr in summary:
        print(f"{v:9} {n:6d} {m:4d} {mc:9.3f} {fr:9.3f}")
    print(f"saved -> {os.path.join(args.out, 'kpts2d.npz')}")


if __name__ == "__main__":
    main()
