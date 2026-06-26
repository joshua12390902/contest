#!/usr/bin/env python
"""Batch-process Scene1 instances reusing ONE fixed calibration (cameras are
fixed within a scene). Per instance: download -> 2D detect -> triangulate with
the fixed P1/P2/P3 -> scale -> BODY_25 keypoints3d JSON. Reports reproj error
per instance (low => fixed-camera assumption holds; reuse is valid)."""
import os, sys, json, argparse, glob
import numpy as np, cv2
from huggingface_hub import hf_hub_download
import requests
from rtmlib import Wholebody

REPO = "Atomathtang/Scene1"
RAW = "/workspace/perceptalign_repro/PerceptAlign/data/raw/Scene1"
VIEWS = ["output1", "output2", "output3"]
NB, CONF, SUB, NGT = 23, 0.45, 5, 30
B25 = [('c',0),('mid',5,6),('c',6),('c',8),('c',10),('c',5),('c',7),('c',9),
       ('mid',11,12),('c',12),('c',14),('c',16),('c',11),('c',13),('c',15),
       ('c',2),('c',1),('c',4),('c',3),('c',17),('c',18),('c',19),('c',20),('c',21),('c',22)]


def hf_list(inst):  # inst like "user1/action1/2-1-1"
    r = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{inst}?recursive=1")
    return [x["path"] for x in r.json() if x["type"] == "file"]


def dl(inst):
    for v in VIEWS:  # mp4 name varies (segment_<N>.mp4) -> list the dir
        r = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{inst}/{v}")
        for x in r.json():
            if x["path"].endswith(".mp4"):
                hf_hub_download(REPO, x["path"], repo_type="dataset", local_dir=RAW)
    for f in [p for p in hf_list(inst) if "/csi_mat/" in p and p.endswith(".mat")]:
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW)


def detect(model, inst_dir):
    XY = np.full((len(VIEWS), NGT, 133, 2), np.nan, np.float32)
    SC = np.zeros((len(VIEWS), NGT, 133), np.float32)
    for vi, v in enumerate(VIEWS):
        mp4 = glob.glob(os.path.join(inst_dir, v, "*.mp4"))
        cap = cv2.VideoCapture(mp4[0] if mp4 else os.path.join(inst_dir, v, "none"))
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % SUB == 0: frames.append(fr)
            i += 1
        cap.release()
        for t, fr in enumerate(frames[:NGT]):
            k, s = model(fr)
            if k.shape[0] == 0: continue
            areas = [(p[:,0].max()-p[:,0].min())*(p[:,1].max()-p[:,1].min()) for p in k]
            j = int(np.argmax(areas)); XY[vi, t], SC[vi, t] = k[j], s[j]
    return XY, SC


def triangulate(XY, SC, Ps):
    P3d = np.full((NGT, NB, 3), np.nan)
    for t in range(NGT):
        for j in range(NB):
            seen = [(Ps[vi], XY[vi, t, j]) for vi in range(3) if SC[vi, t, j] > CONF]
            if len(seen) >= 2:
                A = []
                for P, (x, y) in seen:
                    A.append(x*P[2]-P[0]); A.append(y*P[2]-P[1])
                _, _, Vt = np.linalg.svd(np.array(A)); X = Vt[-1]; P3d[t, j] = X[:3]/X[3]
    return P3d


def reproj_rms(XY, SC, Ps, P3d):
    e = []
    for t in range(NGT):
        for j in range(NB):
            if np.isnan(P3d[t,j,0]): continue
            Xh = np.r_[P3d[t,j],1]
            for vi in range(3):
                if SC[vi,t,j] > CONF:
                    pr = Ps[vi]@Xh; pr = pr[:2]/pr[2]; e.append(np.linalg.norm(pr-XY[vi,t,j]))
    return float(np.sqrt(np.mean(np.square(e)))) if e else np.nan


def write_json(P3d, SC, inst_dir):
    kp_dir = os.path.join(inst_dir, "default", "smplx", "keypoints3d")
    os.makedirs(kp_dir, exist_ok=True)
    def pos(t, s): return P3d[t, s[1]] if s[0]=='c' else (P3d[t,s[1]]+P3d[t,s[2]])/2
    def cf(t, s):
        if s[0]=='c': return float(np.max(SC[:,t,s[1]]))
        return float(min(np.max(SC[:,t,s[1]]), np.max(SC[:,t,s[2]])))
    for t in range(NGT):
        rows = []
        for s in B25:
            p = pos(t, s)
            rows.append([0.,0.,0.,0.] if np.any(np.isnan(p)) else
                        [float(p[0]),float(p[1]),float(p[2]),round(cf(t,s),3)])
        json.dump([{"keypoints3d": rows}], open(os.path.join(kp_dir, f"{t:06d}.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="/workspace/perceptalign_repro/selflabel/out/1-1-1/calib.npz")
    ap.add_argument("--instances", nargs="+", required=True)  # e.g. user1/action1/2-1-1
    args = ap.parse_args()
    cal = np.load(args.calib); Ps = [cal["P1"], cal["P2"], cal["P3"]]; scale = float(cal["scale"])
    model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    print(f"{'instance':22} {'reprojRMS':>9} {'thigh(m)':>9} {'shoulderW':>9}")
    for inst in args.instances:
      try:
        inst_dir = os.path.join(RAW, inst)
        if not glob.glob(os.path.join(inst_dir, "output1", "*.mp4")):
            dl(inst)
        XY, SC = detect(model, inst_dir)
        P3d = triangulate(XY, SC, Ps)
        rms = reproj_rms(XY, SC, Ps, P3d)
        P3d_m = P3d * scale
        write_json(P3d_m, SC, inst_dir)
        th = np.nanmedian(np.linalg.norm(P3d_m[:,12]-P3d_m[:,14],axis=1))
        sw = np.nanmedian(np.linalg.norm(P3d_m[:,5]-P3d_m[:,6],axis=1))
        print(f"{inst:22} {rms:9.2f} {th:9.3f} {sw:9.3f}")
      except Exception as e:
        print(f"{inst:22} SKIP ({type(e).__name__}: {e})")
    print("done")


if __name__ == "__main__":
    main()
