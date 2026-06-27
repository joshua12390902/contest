#!/usr/bin/env python
"""Download ONLY the csi_mat/*.mat files for a list of instances (we already have
the GT in 3DGroundTruth/, so no video / no re-labeling needed). Used to fetch CSI
for Scene1/Scene2 (whose GT is in the repo but CSI is not local) for training.

  python selflabel/scripts/download_csi.py --repo Atomathtang/Scene1 \
      --raw PerceptAlign/data/raw/Scene1 --instances_file subset_scene1.txt --workers 12
"""
import os, time, argparse, requests
from concurrent.futures import ThreadPoolExecutor

TOKEN = os.environ.get("HF_TOKEN"); HDR = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def tree(repo, p, retries=6):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{p}"
    for k in range(retries):
        try:
            r = requests.get(url, headers=HDR, timeout=20); r.raise_for_status(); return r.json()
        except Exception:
            if k == retries - 1: return []
            time.sleep(2 * (k + 1))
    return []


def robust_dl(repo, raw, path):
    dest = os.path.join(raw, path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0: return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    for a in range(4):
        try:
            with requests.get(url, headers=HDR, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                with open(dest + ".part", "wb") as f:
                    for c in r.iter_content(1 << 20): f.write(c)
            os.replace(dest + ".part", dest); return True
        except Exception:
            try: os.remove(dest + ".part")
            except OSError: pass
            time.sleep(2 * (a + 1))
    return False


def fetch_one(repo, raw, inst):
    inst_dir = os.path.join(raw, inst, "csi_mat")
    if os.path.isdir(inst_dir) and len([f for f in os.listdir(inst_dir) if f.endswith(".mat")]) >= 3:
        return "skip"
    mats = [x["path"] for x in tree(repo, f"{inst}/csi_mat") if x["path"].endswith(".mat")]
    if not mats:
        return "err"
    ok = all(robust_dl(repo, raw, m) for m in mats)
    return "ok" if ok else "err"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--instances_file", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    insts = [l.strip() for l in open(args.instances_file) if l.strip()]
    ok = skip = err = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(lambda x: fetch_one(args.repo, args.raw, x), insts)):
            ok += r == "ok"; skip += r == "skip"; err += r == "err"
            if (i + 1) % 50 == 0:
                rate = (i + 1) / max(1, time.time() - t0)
                print(f"[{i+1}/{len(insts)}] ok={ok} skip={skip} err={err} {rate:.1f}/s", flush=True)
    print(f"DONE ok={ok} skip={skip} err={err} of {len(insts)}")


if __name__ == "__main__":
    main()
