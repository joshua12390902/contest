# Scene4 + Scene5:從自製 GT 到訓練(怎麼訓練)

> 接續 `selflabel/` 自標出的 Scene4/5 3D 骨架 GT,把它餵進 PerceptAlign 模型訓練。
> 對照 `docs/TRAIN_PY_EXPLAINED.md`(train.py 流程)與論文 Method。

## 0. 環境(訓練用哪個 python)

自標(rtmlib/onnxruntime)用 `contest/.venv`;**訓練/前處理另用** `sim-evals` venv —— 它已內含
`torch 2.7+cu126 + torchvision 0.22 + h5py + pyyaml`,模型的 `torchvision.resnet34` 直接可用,不必再裝。

```bash
PY=/home/pairlab/sim-evals/.venv/bin/python
cd /home/pairlab/contest/PerceptAlign
```

## 1. 資料流(三段)

```
selflabel/run_full.sh           data/raw/SceneX/<inst>/default/smplx/keypoints3d/*.json   (GT, batch_full 產)
   │                            data/raw/SceneX/<inst>/csi_mat/*-r{1,2,3}.mat              (CSI, 保留)
   ▼
tools/preprocess.py  ──►  data/preprocessed_actions/SceneX/pt/*.pt  +  manifest.json
   │   每個 .pt ≈ 52MB([T=30,Nr=3,3,224,224] CSI 影像 + keypoints + tx/rx)
   ▼
tools/train.py --config configs/*.yaml  ──►  weights/<exp>_best.pth   (依 protocol 切 train/val/test)
```

⚠️ **磁碟**:`.pt` 每個 ~52MB,全量 3,238 個 ≈ 168GB 放不下。**只前處理「每 user 一兩百個」的平衡子集**即可
(論文相對效果實驗「幾百個 instance 足矣」)。用 `selflabel/scripts/make_subset.py` 建 symlink 子集再前處理。

## 2. 前處理(建子集 → preprocess)

```bash
cd /home/pairlab/contest
# (1) 平衡子集:每 user 取 120 個有完整 GT 的 instance(symlink,不佔空間)
.venv/bin/python selflabel/scripts/make_subset.py \
  --scene_root PerceptAlign/data/raw/Scene4 --out PerceptAlign/data/train_subset/Scene4 --per_user 120
.venv/bin/python selflabel/scripts/make_subset.py \
  --scene_root PerceptAlign/data/raw/Scene5 --out PerceptAlign/data/train_subset/Scene5 --per_user 120

# (2) preprocess 兩個 scene 進「同一個」manifest(train.py 靠 manifest 切 protocol)
cd PerceptAlign
PY=/home/pairlab/sim-evals/.venv/bin/python
PYTHONPATH=$(pwd) $PY tools/preprocess.py --scene_root data/train_subset/Scene4 \
  --out_root data/preprocessed_actions --scene_name Scene4 \
  --geometry_json ../selflabel/calibs/geometry_scene4.json
PYTHONPATH=$(pwd) $PY tools/preprocess.py --scene_root data/train_subset/Scene5 \
  --out_root data/preprocessed_actions --scene_name Scene5 \
  --geometry_json ../selflabel/calibs/geometry_scene5.json
```

⚠️ **不要加 `--apply_scene_transform`**:共享 GT(`3DGroundTruth/`)已經是最終 **person-centric** frame
(`normalize_gt.py` 做過:z=離地高度、x=左右髖、y=前後、原點=骨盆),再減 tx 會破壞它。
`--geometry_json` 仍要給 → tx/rx 會寫進 .pt,訓練時 train.py 算 `rel_rx = rx - tx` 當幾何條件。

> 若是從乾淨的 `3DGroundTruth/` 起跑(本機 data/raw 已被 normalize 過,可跳過):
> `python selflabel/scripts/unpack_gt.py --gt_dir 3DGroundTruth/Scene4 --scene_root PerceptAlign/data/raw/Scene4`

## 3. 訓練(三個 protocol)

| config | 切法 | 意義 | 絕對 MPJPE 可信? |
|---|---|---|---|
| `cross_subject_scene4_leave1.yaml` | Scene4:train user2+3 / test user1 | 同場景、未見受試者泛化 | ✅(同一座標系) |
| `cross_subject_scene5_leave1.yaml` | Scene5:同上 | 同上 | ✅ |
| `cross_scene_train4_test5.yaml` | train Scene4 / test Scene5 | 跨場景 | ✅ person-centric frame 跨場景一致(z=離地、x=左右髖)→ 絕對 MPJPE 現在可比;這正是 person-centric 表示法的目的 |

```bash
PYTHONPATH=$(pwd) $PY tools/train.py --config configs/cross_subject_scene4_leave1.yaml --device cuda:0
# log 每 epoch:val MPJPE(mm) / PCK@20 / PCK@50;MPJPE 創新低存 weights/<exp>_best.pth
```
快速驗證可加 `--max_train_batches 30 --max_val_batches 20`(幾分鐘看 loss 是否下降),確認沒問題再全量。

## 4. 核心科學實驗:geometry conditioning 消融(rel_rx 開/關)

論文主張「Breaking Coordinate Overfitting」= 把收發器幾何當**條件**(rel_rx)而非死背座標,能改善跨域。
要做這個消融需「同一份 GT、只切換有沒有把 rel_rx 餵進模型」。train.py 目前**只要 .pt 有 tx/rx 就一定用 rel_rx**,
沒有關閉開關 → 乾淨消融需在 train.py 加一個 `model.use_geometry`(false 時把 rel_rx 傳 None)。
這是緊接的下一步(見 README「已知限制」:能復現的就是這個相對效果)。

## 5. 評估

```bash
PYTHONPATH=$(pwd) $PY tools/eval.py --config configs/cross_subject_scene4_leave1.yaml \
  --checkpoint weights/cross_subject_scene4_leave1_best.pth --split test --device cuda:0
```
> 注意 eval 的 MPJPE 是裸 L2、無 Procrustes/root-align,且 GT 在自製座標系 → **不可對比論文 Table 3**,
> 只用於「同設定下開/關 geometry、跨 user/scene 的相對退化」比較。
