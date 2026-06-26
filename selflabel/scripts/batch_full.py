#!/usr/bin/env python
"""Production batch: self-label an ENTIRE scene reusing one fixed calibration.
GPU 2D detection, auto-QC (reject implausible reconstructions), resumable
(skips done instances), deletes videos after GT extraction to save disk."""
import os, sys, json, glob, time, argparse
import numpy as np, cv2, requests
from huggingface_hub import hf_hub_download
from rtmlib import Wholebody

REPO = os.environ.get("SCENE_REPO", "Atomathtang/Scene1")
RAW = os.environ.get("SCENE_RAW", "/workspace/perceptalign_repro/PerceptAlign/data/raw/Scene1")
TOKEN = os.environ.get("HF_TOKEN")
HDR = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
VIEWS = ["output1", "output2", "output3"]
NB, CONF, SUB, NGT = 23, 0.45, 5, 30
B25 = [('c',0),('mid',5,6),('c',6),('c',8),('c',10),('c',5),('c',7),('c',9),
       ('mid',11,12),('c',12),('c',14),('c',16),('c',11),('c',13),('c',15),
       ('c',2),('c',1),('c',4),('c',3),('c',17),('c',18),('c',19),('c',20),('c',21),('c',22)]
# QC gates
QC_REPROJ_PX, QC_THIGH, QC_SHW, QC_FILL = 15.0, (0.28, 0.62), (0.24, 0.56), 0.5


def tree(p):
    return requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{p}",
                        headers=HDR, timeout=20).json()


def robust_dl(path):
    """Stream-download one repo file with timeout+retry (avoids HF CDN stalls)."""
    dest = os.path.join(RAW, path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    for attempt in range(4):
        try:
            with requests.get(url, headers=HDR, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                with open(dest + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(dest + ".part", dest)
            return True
        except Exception:
            try: os.remove(dest + ".part")
            except OSError: pass
            time.sleep(2 * (attempt + 1))
    return False


def dl(inst):
    for v in VIEWS:
        for x in tree(f"{inst}/{v}"):
            if x["path"].endswith(".mp4"):
                robust_dl(x["path"])
    for x in tree(f"{inst}/csi_mat"):
        if x["path"].endswith(".mat"):
            robust_dl(x["path"])


def detect(model, inst_dir):
    XY = np.full((3, NGT, 133, 2), np.nan, np.float32); SC = np.zeros((3, NGT, 133), np.float32)
    for vi, v in enumerate(VIEWS):
        mp4 = glob.glob(os.path.join(inst_dir, v, "*.mp4"))
        if not mp4: continue
        cap = cv2.VideoCapture(mp4[0]); frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % SUB == 0: frames.append(fr)
            i += 1
        cap.release()
        for t, fr in enumerate(frames[:NGT]):
            k, s = model(fr)
            if k.shape[0] == 0: continue
            a = [(p[:,0].max()-p[:,0].min())*(p[:,1].max()-p[:,1].min()) for p in k]
            j = int(np.argmax(a)); XY[vi,t], SC[vi,t] = k[j], s[j]
    return XY, SC


def triangulate(XY, SC, Ps):
    P3d = np.full((NGT, NB, 3), np.nan)
    for t in range(NGT):
        for j in range(NB):
            seen = [(Ps[vi], XY[vi,t,j]) for vi in range(3) if SC[vi,t,j] > CONF]
            if len(seen) >= 2:
                A = []
                for P,(x,y) in seen: A.append(x*P[2]-P[0]); A.append(y*P[2]-P[1])
                _,_,Vt = np.linalg.svd(np.array(A)); X = Vt[-1]; P3d[t,j] = X[:3]/X[3]
    return P3d


def reproj_rms(XY, SC, Ps, P3d):
    e = []
    for t in range(NGT):
        for j in range(NB):
            if np.isnan(P3d[t,j,0]): continue
            Xh = np.r_[P3d[t,j],1]
            for vi in range(3):
                if SC[vi,t,j] > CONF:
                    pr = Ps[vi]@Xh; e.append(np.linalg.norm(pr[:2]/pr[2]-XY[vi,t,j]))
    return float(np.sqrt(np.mean(np.square(e)))) if e else np.inf


def write_json(P3d, SC, inst_dir):
    kp_dir = os.path.join(inst_dir, "default", "smplx", "keypoints3d"); os.makedirs(kp_dir, exist_ok=True)
    def pos(t,s): return P3d[t,s[1]] if s[0]=='c' else (P3d[t,s[1]]+P3d[t,s[2]])/2
    def cf(t,s):
        return float(np.max(SC[:,t,s[1]])) if s[0]=='c' else float(min(np.max(SC[:,t,s[1]]),np.max(SC[:,t,s[2]])))
    for t in range(NGT):
        rows = [[0.,0.,0.,0.] if np.any(np.isnan(pos(t,s))) else
                [float(pos(t,s)[0]),float(pos(t,s)[1]),float(pos(t,s)[2]),round(cf(t,s),3)] for s in B25]
        json.dump([{"keypoints3d": rows}], open(os.path.join(kp_dir, f"{t:06d}.json"), "w"))


def ensure_dl(inst):
    inst_dir = os.path.join(RAW, inst)
    if glob.glob(os.path.join(inst_dir, "output1", "*.mp4")):
        return True
    try:
        dl(inst); return True
    except Exception:
        return False


def main():
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances_file", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--workers", type=int, default=16)   # parallel download threads
    ap.add_argument("--window", type=int, default=64)    # prefetch depth
    args = ap.parse_args()
    cal = np.load(args.calib); Ps = [cal["P1"], cal["P2"], cal["P3"]]; scale = float(cal["scale"])
    model = Wholebody(mode="balanced", backend="onnxruntime", device="cuda")
    allinsts = [l.strip() for l in open(args.instances_file) if l.strip()]

    def is_done(inst):
        kp = os.path.join(RAW, inst, "default", "smplx", "keypoints3d")
        return os.path.exists(kp) and len(glob.glob(os.path.join(kp, "*.json"))) >= NGT
    pending = [i for i in allinsts if not is_done(i)]
    skip = len(allinsts) - len(pending)
    lg = open(args.log, "a"); t0 = time.time(); acc = rej = err = 0
    print(f"pending={len(pending)} skip={skip}", flush=True)

    # parallel-prefetch downloads, serial GPU processing in list order
    ex = ThreadPoolExecutor(max_workers=args.workers)
    futs = {}
    def submit(k):
        if 0 <= k < len(pending): futs[k] = ex.submit(ensure_dl, pending[k])
    for k in range(min(args.window, len(pending))): submit(k)

    for k in range(len(pending)):
        inst = pending[k]; inst_dir = os.path.join(RAW, inst)
        try:
            futs[k].result()                       # wait until this instance downloaded
            XY, SC = detect(model, inst_dir)
            P3d = triangulate(XY, SC, Ps)
            rms = reproj_rms(XY, SC, Ps, P3d)
            P3d_m = P3d * scale
            thigh = np.nanmedian(np.linalg.norm(P3d_m[:,12]-P3d_m[:,14],axis=1))
            shw = np.nanmedian(np.linalg.norm(P3d_m[:,5]-P3d_m[:,6],axis=1))
            fill = np.mean(~np.isnan(P3d_m[...,0]))
            ok = (rms < QC_REPROJ_PX and QC_THIGH[0]<thigh<QC_THIGH[1]
                  and QC_SHW[0]<shw<QC_SHW[1] and fill > QC_FILL)
            if ok: write_json(P3d_m, SC, inst_dir); acc += 1; st = "ACC"
            else: rej += 1; st = "REJ"
            lg.write(f"{inst}\t{st}\trms={rms:.1f}\tthigh={thigh:.2f}\tshw={shw:.2f}\tfill={fill:.2f}\n")
            for v in VIEWS:
                for f in glob.glob(os.path.join(inst_dir, v, "*.mp4")): os.remove(f)
        except Exception as e:
            err += 1; lg.write(f"{inst}\tERR\t{type(e).__name__}:{str(e)[:80]}\n")
        finally:
            futs.pop(k, None); submit(k + args.window)
        if (k+1) % 25 == 0:
            lg.flush()
            rate = (k+1)/max(1, time.time()-t0)
            print(f"[{k+1}/{len(pending)}] acc={acc} rej={rej} err={err} "
                  f"{rate:.2f} inst/s ETA {(len(pending)-k-1)/max(rate,0.01)/3600:.1f}h", flush=True)
    done = skip
    lg.write(f"# SUMMARY acc={acc} rej={rej} err={err} skip={done} of {len(insts)}\n"); lg.flush()
    print(f"DONE acc={acc} rej={rej} err={err} skip={done}", flush=True)


if __name__ == "__main__":
    main()
