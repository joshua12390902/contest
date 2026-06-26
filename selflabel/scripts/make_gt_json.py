#!/usr/bin/env python
"""Step D: convert metric 3D skeleton (COCO-23) to BODY_25 and write the repo's
keypoints3d JSON schema, one file per frame. Also dump a reprojection overlay
for visual QC.
"""
import os, json, argparse
import numpy as np, cv2

VIEWS = ["output1", "output2", "output3"]
# BODY_25 idx -> source. ('c',i)=coco joint i ; ('mid',a,b)=midpoint of coco a,b
B25 = [('c',0),('mid',5,6),('c',6),('c',8),('c',10),('c',5),('c',7),('c',9),
       ('mid',11,12),('c',12),('c',14),('c',16),('c',11),('c',13),('c',15),
       ('c',2),('c',1),('c',4),('c',3),('c',17),('c',18),('c',19),('c',20),('c',21),('c',22)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst_out", required=True)
    ap.add_argument("--instance", required=True)   # where to write default/smplx/keypoints3d
    args = ap.parse_args()

    P3d = np.load(os.path.join(args.inst_out, "skel3d_metric.npy"))   # [T,23,3]
    cal = np.load(os.path.join(args.inst_out, "calib.npz"))
    kp = np.load(os.path.join(args.inst_out, "kpts2d.npz"))
    T = P3d.shape[0]
    sc = {v: kp[f"{v}_sc"] for v in VIEWS}

    kp_dir = os.path.join(args.instance, "default", "smplx", "keypoints3d")
    os.makedirs(kp_dir, exist_ok=True)

    def jconf(t, src):
        if src[0] == 'c':
            return float(np.max([sc[v][t, src[1]] for v in VIEWS]))
        return float(min(np.max([sc[v][t, src[1]] for v in VIEWS]),
                         np.max([sc[v][t, src[2]] for v in VIEWS])))

    def jpos(t, src):
        if src[0] == 'c':
            return P3d[t, src[1]]
        return (P3d[t, src[1]] + P3d[t, src[2]]) / 2

    for t in range(T):
        rows = []
        for src in B25:
            p = jpos(t, src); c = jconf(t, src)
            if np.any(np.isnan(p)):
                rows.append([0.0, 0.0, 0.0, 0.0])
            else:
                rows.append([float(p[0]), float(p[1]), float(p[2]), round(c, 3)])
        json.dump([{"keypoints3d": rows}], open(os.path.join(kp_dir, f"{t:06d}.json"), "w"))
    print(f"[json] wrote {T} BODY_25 keypoints3d files -> {kp_dir}")

    # --- reprojection overlay for visual QC (view1, middle frame) ---
    P1 = cal["P1"]; scale = float(cal["scale"])
    LINKS = [(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(8,12),
             (9,10),(10,11),(12,13),(13,14),(0,1)]
    t = 15
    cap = cv2.VideoCapture(os.path.join(args.instance, "output1", "segment_1.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t * 5); ok, frame = cap.read(); cap.release()
    pts2d = {}
    for bi, src in enumerate(B25):
        p = jpos(t, src)
        if np.any(np.isnan(p)): continue
        Xh = np.r_[p, 1]; pr = P1 @ Xh; pts2d[bi] = (pr[:2] / pr[2])
    for a, b in LINKS:
        if a in pts2d and b in pts2d:
            cv2.line(frame, tuple(pts2d[a].astype(int)), tuple(pts2d[b].astype(int)), (0, 255, 0), 3)
    for bi, p in pts2d.items():
        cv2.circle(frame, tuple(p.astype(int)), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(args.inst_out, "debug", "reproj3d_view1.png"), frame)
    print(f"[qc] reprojection overlay -> {args.inst_out}/debug/reproj3d_view1.png")


if __name__ == "__main__":
    main()
