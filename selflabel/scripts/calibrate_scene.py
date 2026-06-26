#!/usr/bin/env python
"""Turnkey per-scene calibration: given ONE reference instance, self-calibrate
the 3 cameras from the person's 2D keypoints (essential matrix + PnP), and emit
calib.npz (P1,P2,P3,scale,K) + a geometry_config.json template (scene_matrix=I,
tx/rx auto-placed on the floor plane — refine later; skeleton GT does not need
tx/rx, only preprocess --apply_scene_transform does).

Run ONCE per scene; Scene3 needs one run per layout (A=user1/2,B=3/4,C=5/6).
"""
import os, sys, json, glob, time, argparse
import numpy as np, cv2, requests
from rtmlib import Wholebody

TOKEN = os.environ.get("HF_TOKEN"); HDR = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
VIEWS = ["output1", "output2", "output3"]
NB, CONF, NGT = 23, 0.45, 30
K = np.array([[1380, 0, 960], [0, 1380, 540], [0, 0, 1]], float)


def robust_dl(repo, raw, path):
    dest = os.path.join(raw, path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0: return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    for a in range(4):
        try:
            with requests.get(url, headers=HDR, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                with open(dest + ".part", "wb") as f:
                    for c in r.iter_content(1 << 20): f.write(c)
            os.replace(dest + ".part", dest); return
        except Exception:
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"download failed: {path}")


def tree(repo, p, retries=8):
    for k in range(retries):
        try:
            r = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main/{p}", headers=HDR, timeout=30)
            r.raise_for_status(); return r.json()
        except Exception:
            if k == retries - 1: raise
            time.sleep(2 * (k + 1))


def adaptive_frames(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 150
    step = max(1, round(total / NGT))
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0: frames.append(fr)
        i += 1
    return frames[:NGT]


def detect(model, inst_dir):
    XY = np.full((3, NGT, 133, 2), np.nan, np.float32); SC = np.zeros((3, NGT, 133), np.float32)
    for vi, v in enumerate(VIEWS):
        mp4 = glob.glob(os.path.join(inst_dir, v, "*.mp4"))
        if not mp4: continue
        cap = cv2.VideoCapture(mp4[0]); frames = adaptive_frames(cap); cap.release()
        for t, fr in enumerate(frames):
            k, s = model(fr)
            if k.shape[0] == 0: continue
            a = [(p[:,0].max()-p[:,0].min())*(p[:,1].max()-p[:,1].min()) for p in k]
            j = int(np.argmax(a)); XY[vi,t], SC[vi,t] = k[j], s[j]
    return XY, SC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--raw", required=True)        # local scene root e.g. .../data/raw/Scene2
    ap.add_argument("--ref", required=True)        # reference instance user1/action1/1-1-1
    ap.add_argument("--out_calib", required=True)
    ap.add_argument("--out_geom", required=True)
    args = ap.parse_args()

    inst_dir = os.path.join(args.raw, args.ref)
    if not glob.glob(os.path.join(inst_dir, "output1", "*.mp4")):
        for v in VIEWS:
            for x in tree(args.repo, f"{args.ref}/{v}"):
                if x["path"].endswith(".mp4"): robust_dl(args.repo, args.raw, x["path"])
    model = Wholebody(mode="balanced", backend="onnxruntime", device="cuda")
    XY, SC = detect(model, inst_dir)

    # essential matrix view1<->view3
    def corr(a, b):
        pa, pb = [], []
        for t in range(NGT):
            for j in range(NB):
                if SC[a,t,j] > CONF and SC[b,t,j] > CONF:
                    pa.append(XY[a,t,j]); pb.append(XY[b,t,j])
        return np.array(pa), np.array(pb)
    pa, pb = corr(0, 2)
    E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R13, t13, mask = cv2.recoverPose(E, pa, pb, K, mask=mask.copy())
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))]); P3 = K @ np.hstack([R13, t13])
    m = mask.ravel() > 0
    X4 = cv2.triangulatePoints(P1, P3, pa[m].T, pb[m].T); Xa = (X4[:3]/X4[3]).T
    # anchor map for PnP
    anchor = {}; idx = [(t,j) for t in range(NGT) for j in range(NB) if SC[0,t,j]>CONF and SC[2,t,j]>CONF]
    for (tj, X) in zip([idx[i] for i in range(len(idx)) if m[i]], Xa): anchor[tj] = X
    obj = [X for tj,X in anchor.items() if SC[1,tj[0],tj[1]]>CONF]
    img = [XY[1,tj[0],tj[1]] for tj,X in anchor.items() if SC[1,tj[0],tj[1]]>CONF]
    _, rvec, tvec = cv2.solvePnP(np.array(obj), np.array(img), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    P2 = K @ np.hstack([cv2.Rodrigues(rvec)[0], tvec]); Ps = [P1, P2, P3]

    # full triangulation + scale + reproj QC
    P3d = np.full((NGT, NB, 3), np.nan)
    for t in range(NGT):
        for j in range(NB):
            seen = [(Ps[vi], XY[vi,t,j]) for vi in range(3) if SC[vi,t,j] > CONF]
            if len(seen) >= 2:
                A = []
                for P,(x,y) in seen: A.append(x*P[2]-P[0]); A.append(y*P[2]-P[1])
                _,_,Vt = np.linalg.svd(np.array(A)); X = Vt[-1]; P3d[t,j] = X[:3]/X[3]
    sc_a = (P3d[:,5]+P3d[:,6])/2; ac = (P3d[:,15]+P3d[:,16])/2
    scale = 1.30 / np.nanmedian(np.linalg.norm(sc_a-ac, axis=1))
    P3d_m = P3d * scale
    errs = []
    for t in range(NGT):
        for j in range(NB):
            if np.isnan(P3d[t,j,0]): continue
            Xh = np.r_[P3d[t,j],1]
            for vi in range(3):
                if SC[vi,t,j] > CONF:
                    pr = Ps[vi]@Xh; errs.append(np.linalg.norm(pr[:2]/pr[2]-XY[vi,t,j]))
    rms = float(np.sqrt(np.mean(np.square(errs))))
    thigh = np.nanmedian(np.linalg.norm(P3d_m[:,12]-P3d_m[:,14],axis=1))

    np.savez(args.out_calib, P1=P1, P2=P2, P3=P3, scale=scale, K=K)
    # floor plane -> auto tx/rx (placeholder, refine later)
    feet = P3d_m[:,17:23,:].reshape(-1,3); feet = feet[~np.isnan(feet).any(1)]
    c = feet.mean(0); _,_,Vt = np.linalg.svd(feet-c); n = Vt[-1]
    e1 = Vt[0]-n*(n@Vt[0]); e1/=np.linalg.norm(e1); e2 = np.cross(n,e1)
    fp = lambda a,b: (c+a*e1+b*e2).tolist()
    geom = {"scene": os.path.basename(args.raw), "tx": fp(-2,0.5),
            "rx": [fp(2,0.8), fp(0,-2), fp(-1.8,1.2)],
            "scene_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            "_note": "tx/rx are AUTO floor-plane placeholders; refine by locating real antennas. Model tolerates tx/rx error well."}
    json.dump(geom, open(args.out_geom, "w"), indent=2)
    print(f"[{args.repo}] ref={args.ref}  reproj={rms:.2f}px  thigh={thigh:.2f}m  scale={scale:.3f}")
    print(f"  -> {args.out_calib} , {args.out_geom}")
    print(f"  {'OK' if rms<10 and 0.3<thigh<0.6 else 'CHECK!'}: 校正品質{'良好' if rms<10 else '需檢查(換參考 instance)'}")


if __name__ == "__main__":
    main()
