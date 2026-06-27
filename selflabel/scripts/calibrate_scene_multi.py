#!/usr/bin/env python
"""Robust per-scene calibration from MULTIPLE reference instances.

calibrate_scene.py fits the view1<->view3 essential matrix to ONE instance's
human joints, which span only one floor location. When that is poorly
conditioned (e.g. Scene3), the recovered camera geometry overfits to that
location and reprojects badly elsewhere (60-80px at other locations).

This version pools 2D correspondences across several reference instances chosen
to span the room (ideally one per location L in L-O-R), so the essential matrix
is constrained globally -> a single calib valid at every location.

  python selflabel/scripts/calibrate_scene_multi.py --repo Atomathtang/Scene3 \
    --raw PerceptAlign/data/raw/Scene3 \
    --refs user1/action1/1-1-1,user1/action1/2-1-1,user1/action1/3-1-1,user1/action1/4-1-1,user1/action1/5-1-1 \
    --out_calib selflabel/calibs/calib_scene3_A.npz --out_geom selflabel/calibs/geometry_scene3_A.json
"""
import os, json, glob, time, argparse
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
    ap.add_argument("--raw", required=True)
    ap.add_argument("--refs", required=True, help="comma-separated instances spanning locations")
    ap.add_argument("--out_calib", required=True)
    ap.add_argument("--out_geom", required=True)
    ap.add_argument("--dry_run", action="store_true", help="print QC, do not write calib/geom")
    args = ap.parse_args()
    refs = [r.strip() for r in args.refs.split(",") if r.strip()]

    model = Wholebody(mode="balanced", backend="onnxruntime", device="cuda")
    dets = []  # per-ref (XY, SC)
    for ref in refs:
        inst_dir = os.path.join(args.raw, ref)
        if not glob.glob(os.path.join(inst_dir, "output1", "*.mp4")):
            for v in VIEWS:
                for x in tree(args.repo, f"{ref}/{v}"):
                    if x["path"].endswith(".mp4"): robust_dl(args.repo, args.raw, x["path"])
        dets.append(detect(model, inst_dir))
        print(f"  detected {ref}")

    # ---- pool view1<->view3 correspondences across ALL refs ----
    pa, pb = [], []
    for XY, SC in dets:
        for t in range(NGT):
            for j in range(NB):
                if SC[0,t,j] > CONF and SC[2,t,j] > CONF:
                    pa.append(XY[0,t,j]); pb.append(XY[2,t,j])
    pa, pb = np.array(pa), np.array(pb)
    E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.9999, threshold=1.0)
    _, R13, t13, mask = cv2.recoverPose(E, pa, pb, K, mask=mask.copy())
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))]); P3 = K @ np.hstack([R13, t13])

    # ---- PnP view2 from pooled 3D anchors (triangulate each ref's pts, gather) ----
    obj, img = [], []
    for XY, SC in dets:
        for t in range(NGT):
            for j in range(NB):
                if SC[0,t,j] > CONF and SC[2,t,j] > CONF and SC[1,t,j] > CONF:
                    X4 = cv2.triangulatePoints(P1, P3, XY[0,t,j].reshape(2,1), XY[2,t,j].reshape(2,1))
                    X = (X4[:3]/X4[3]).ravel()
                    if np.isfinite(X).all():
                        obj.append(X); img.append(XY[1,t,j])
    _, rvec, tvec = cv2.solvePnP(np.array(obj), np.array(img), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    P2 = K @ np.hstack([cv2.Rodrigues(rvec)[0], tvec]); Ps = [P1, P2, P3]

    # ---- scale from shoulder<->ankle (~1.30m), reproj QC per location ----
    def triang(XY, SC):
        P3d = np.full((NGT, NB, 3), np.nan)
        for t in range(NGT):
            for j in range(NB):
                seen = [(Ps[vi], XY[vi,t,j]) for vi in range(3) if SC[vi,t,j] > CONF]
                if len(seen) >= 2:
                    A = []
                    for P,(x,y) in seen: A.append(x*P[2]-P[0]); A.append(y*P[2]-P[1])
                    _,_,Vt = np.linalg.svd(np.array(A)); X = Vt[-1]; P3d[t,j] = X[:3]/X[3]
        return P3d
    scales = []
    for XY, SC in dets:
        P3d = triang(XY, SC)
        sc_a = (P3d[:,5]+P3d[:,6])/2; ac = (P3d[:,15]+P3d[:,16])/2
        scales.append(1.30 / np.nanmedian(np.linalg.norm(sc_a-ac, axis=1)))
    scale = float(np.nanmedian(scales))

    print(f"\n[{args.repo}] refs={len(refs)}  scale={scale:.3f}")
    print(f"{'ref':28s} reproj(px)  thigh(m)")
    all_ok = True
    last_P3d_m = None
    for ref, (XY, SC) in zip(refs, dets):
        P3d = triang(XY, SC); P3d_m = P3d * scale; last_P3d_m = P3d_m
        errs = []
        for t in range(NGT):
            for j in range(NB):
                if np.isnan(P3d[t,j,0]): continue
                Xh = np.r_[P3d[t,j],1]
                for vi in range(3):
                    if SC[vi,t,j] > CONF:
                        pr = Ps[vi]@Xh; errs.append(np.linalg.norm(pr[:2]/pr[2]-XY[vi,t,j]))
        rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float('inf')
        thigh = np.nanmedian(np.linalg.norm(P3d_m[:,12]-P3d_m[:,14],axis=1))
        ok = rms < 15 and 0.28 < thigh < 0.62
        all_ok = all_ok and ok
        print(f"{ref:28s} {rms:8.1f}   {thigh:6.2f}   {'OK' if ok else 'BAD'}")

    if args.dry_run:
        print(f"\nDRY RUN — overall {'PASS' if all_ok else 'FAIL'} (not written)")
        return

    np.savez(args.out_calib, P1=P1, P2=P2, P3=P3, scale=scale, K=K)
    feet = last_P3d_m[:,17:23,:].reshape(-1,3); feet = feet[~np.isnan(feet).any(1)]
    c = feet.mean(0); _,_,Vt = np.linalg.svd(feet-c); n = Vt[-1]
    e1 = Vt[0]-n*(n@Vt[0]); e1/=np.linalg.norm(e1); e2 = np.cross(n,e1)
    fp = lambda a,b: (c+a*e1+b*e2).tolist()
    geom = {"scene": os.path.basename(args.raw), "tx": fp(-2,0.5),
            "rx": [fp(2,0.8), fp(0,-2), fp(-1.8,1.2)],
            "scene_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            "_note": "tx/rx are AUTO floor-plane placeholders; refine by locating real antennas."}
    json.dump(geom, open(args.out_geom, "w"), indent=2)
    print(f"\noverall {'PASS' if all_ok else 'FAIL'} -> wrote {args.out_calib}, {args.out_geom}")


if __name__ == "__main__":
    main()
