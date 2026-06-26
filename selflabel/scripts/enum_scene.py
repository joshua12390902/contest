#!/usr/bin/env python
"""Enumerate all instance dirs (user*/action*/<L-O-R>) of a HuggingFace scene
dataset into a text file. Robust to transient network/DNS failures (retries)."""
import os, sys, time, argparse, requests

TOKEN = os.environ.get("HF_TOKEN")
HDR = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def tree(repo, p, retries=6):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{p}"
    for k in range(retries):
        try:
            r = requests.get(url, headers=HDR, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)         # e.g. atomathtang11/Scene2
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    users = sorted([x["path"].split("/")[-1] for x in tree(args.repo, "")
                    if x["type"] == "directory" and "user" in x["path"]],
                   key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)
    insts = []
    for u in users:
        for a in [x["path"].split("/")[-1] for x in tree(args.repo, u) if x["type"] == "directory"]:
            for x in tree(args.repo, f"{u}/{a}"):
                if x["type"] == "directory":
                    insts.append(f"{u}/{a}/{x['path'].split('/')[-1]}")
        print(f"  ...{u} done, running total={len(insts)}", flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(insts))
    print(f"{args.repo}: users={len(users)} instances={len(insts)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
