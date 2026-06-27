#!/usr/bin/env bash
# Full self-labeling run for one or more scenes (reuses each scene's fixed calib).
# Requires HF_TOKEN in env (never hardcode it here). Resumable: skips done instances.
#   export HF_TOKEN=hf_xxx
#   nohup bash selflabel/run_full.sh scene4 scene5 > selflabel/logs/run.out 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source selflabel/env.sh >/dev/null
: "${HF_TOKEN:?HF_TOKEN not set}"
LOGDIR=selflabel/logs; mkdir -p "$LOGDIR"
WORKERS=${WORKERS:-12}; WINDOW=${WINDOW:-64}

run_scene () {  # repo raw calib list tag
  local repo=$1 raw=$2 calib=$3 list=$4 tag=$5
  export SCENE_REPO=$repo SCENE_RAW=$raw
  echo "=== $(date '+%F %T') START $tag  repo=$repo  list=$list ($(wc -l <"$list") inst) ==="
  "$PY" selflabel/scripts/batch_full.py --instances_file "$list" --calib "$calib" \
        --log "$LOGDIR/$tag.log" --workers "$WORKERS" --window "$WINDOW"
  echo "=== $(date '+%F %T') END $tag ==="
}

for s in "$@"; do
  case "$s" in
    scene4) run_scene atomathtang11/Scene4 PerceptAlign/data/raw/Scene4 \
              selflabel/calibs/calib_scene4.npz selflabel/scene4_instances.txt scene4 ;;
    scene5) run_scene atomathtang11/Scene5 PerceptAlign/data/raw/Scene5 \
              selflabel/calibs/calib_scene5.npz selflabel/scene5_instances.txt scene5 ;;
    scene3A) run_scene Atomathtang/Scene3 PerceptAlign/data/raw/Scene3 \
              selflabel/calibs/calib_scene3_A.npz selflabel/scene3A_instances.txt scene3A ;;
    scene3B) run_scene Atomathtang/Scene3 PerceptAlign/data/raw/Scene3 \
              selflabel/calibs/calib_scene3_B.npz selflabel/scene3B_instances.txt scene3B ;;
    scene3C) run_scene Atomathtang/Scene3 PerceptAlign/data/raw/Scene3 \
              selflabel/calibs/calib_scene3_C.npz selflabel/scene3C_instances.txt scene3C ;;
    *) echo "unknown scene tag: $s (supported: scene4 scene5 scene3A scene3B scene3C)"; exit 2 ;;
  esac
done
echo "ALL DONE: $*"
