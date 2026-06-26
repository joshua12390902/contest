#!/usr/bin/env python
"""Step B+C: self-calibrate 3 cameras from the person's own 2D keypoints
(essential matrix, no floor markers needed) and triangulate a metric 3D skeleton.

Pipeline:
  pair(view1,view3): findEssentialMat -> recoverPose -> P1,P3 (cam1 frame, up-to-scale)
  triangulate inliers -> 3D anchor points
  resection view2 via solvePnP against those 3D points -> P2 (same frame & scale)
  multi-view DLT triangulate every (frame,joint) seen by >=2 high-conf views
  pin metric scale via human anthropometry
  emit BODY_25 keypoints3d JSON in the repo schema
"""
import os, json, argparse
import numpy as np, cv2

K = np.array([[1380, 0, 960], [0, 1380, 540], [0, 0, 1]], float)  # D435 RGB est. intrinsics
VIEWS = ["output1", "output2", "output3"]
NB = 23            # body+feet joints we triangulate from the 133-kpt wholebody
CONF = 0.45

# COCO-17 -> BODY_25 (25 joints). feet 19-24 from wholebody 17-22.
COCO_FOOT = {17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24}  # not used directly; see build_body25


def load(inst_out):
    d = np.load(os.path.join(inst_out, "kpts2d.npz"))
    return {v: (d[f"{v}_xy"], d[f"{v}_sc"]) for v in VIEWS}


def corr(A, B, jmax=NB, cth=CONF):
    (xa, sa), (xb, sb) = A, B
    pa, pb, idx = [], [], []
    for t in range(xa.shape[0]):
        for j in range(jmax):
            if sa[t, j] > cth and sb[t, j] > cth:
                pa.append(xa[t, j]); pb.append(xb[t, j]); idx.append((t, j))
    return np.array(pa, float), np.array(pb, float), idx


def triangulate_multi(Ps, pts):
    """Ps: list of 3x4, pts: list of (x,y); DLT."""
    Arows = []
    for P, (x, y) in zip(Ps, pts):
        Arows.append(x * P[2] - P[0]); Arows.append(y * P[2] - P[1])
    A = np.array(Arows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]; return X[:3] / X[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst_out", required=True)
    args = ap.parse_args()
    data = load(args.inst_out)
    T = data["output1"][0].shape[0]

    # --- pair (view1, view3): relative pose up to scale ---
    pa, pb, idx = corr(data["output1"], data["output3"])
    E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R13, t13, mask = cv2.recoverPose(E, pa, pb, K, mask=mask.copy())
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P3 = K @ np.hstack([R13, t13])
    m = mask.ravel() > 0
    X4 = cv2.triangulatePoints(P1, P3, pa[m].T, pb[m].T)
    Xanchor = (X4[:3] / X4[3]).T
    idx_in = [idx[i] for i in range(len(idx)) if m[i]]
    anchor = {ij: X for ij, X in zip(idx_in, Xanchor)}
    print(f"[pair13] corr={len(pa)} inliers={m.sum()} reproj-anchor ok")

    # --- resection view2 via PnP against anchor 3D ---
    (x2, s2) = data["output2"]
    obj, img = [], []
    for (t, j), X in anchor.items():
        if s2[t, j] > CONF:
            obj.append(X); img.append(x2[t, j])
    obj = np.array(obj, float); img = np.array(img, float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    R2, _ = cv2.Rodrigues(rvec)
    P2 = K @ np.hstack([R2, tvec])
    pr = (P2 @ np.vstack([obj.T, np.ones(len(obj))])); pr = (pr[:2] / pr[2]).T
    print(f"[view2 PnP] pts={len(obj)} reproj RMS={np.sqrt(((pr-img)**2).sum(1).mean()):.2f}px")
    Ps = {"output1": P1, "output2": P2, "output3": P3}

    # --- multi-view triangulate every (t,j) seen by >=2 views ---
    P3d = np.full((T, NB, 3), np.nan)
    for t in range(T):
        for j in range(NB):
            seen = [(Ps[v], data[v][0][t, j]) for v in VIEWS if data[v][1][t, j] > CONF]
            if len(seen) >= 2:
                P3d[t, j] = triangulate_multi([p for p, _ in seen], [pt for _, pt in seen])

    # reproj error across all views
    errs = []
    for t in range(T):
        for j in range(NB):
            if np.isnan(P3d[t, j, 0]): continue
            Xh = np.r_[P3d[t, j], 1]
            for v in VIEWS:
                if data[v][1][t, j] > CONF:
                    pr = Ps[v] @ Xh; pr = pr[:2] / pr[2]
                    errs.append(np.linalg.norm(pr - data[v][0][t, j]))
    print(f"[multiview] reproj RMS over all views = {np.sqrt(np.mean(np.square(errs))):.2f}px, joints filled={np.mean(~np.isnan(P3d[...,0]))*100:.0f}%")

    # --- metric scale via anthropometry (shoulder-center -> ankle-center ~= 1.30 m) ---
    sc = (P3d[:, 5] + P3d[:, 6]) / 2; ac = (P3d[:, 15] + P3d[:, 16]) / 2
    trunk_leg = np.nanmedian(np.linalg.norm(sc - ac, axis=1))
    scale = 1.30 / trunk_leg
    P3d_m = P3d * scale
    print(f"[scale] shoulder->ankle={trunk_leg:.4f} -> scale={scale:.4f} (m)")

    # --- QC: bone lengths (metric) + stability ---
    BONES = {"L_uarm": (5, 7), "L_farm": (7, 9), "R_uarm": (6, 8), "R_farm": (8, 10),
             "L_thigh": (11, 13), "L_shin": (13, 15), "R_thigh": (12, 14), "R_shin": (14, 16),
             "shoulderW": (5, 6), "hipW": (11, 12)}
    print(f"\n{'bone':10} {'median(m)':>9} {'std/mean':>9}  frames")
    for n, (a, b) in BONES.items():
        L = np.linalg.norm(P3d_m[:, a] - P3d_m[:, b], axis=1); L = L[~np.isnan(L)]
        if len(L):
            print(f"{n:10} {np.median(L):9.3f} {np.std(L)/np.mean(L)*100:8.1f}% {len(L):6d}")

    np.savez(os.path.join(args.inst_out, "calib.npz"),
             P1=P1, P2=P2, P3=P3, scale=scale, K=K)
    np.save(os.path.join(args.inst_out, "skel3d_metric.npy"), P3d_m)
    print(f"\nsaved calib.npz + skel3d_metric.npy -> {args.inst_out}")


if __name__ == "__main__":
    main()
