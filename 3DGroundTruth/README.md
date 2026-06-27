# 3D GroundTruth — 自製 WiFi CSI 骨架真值（分 Scene1–5）

每個 instance 一個 JSON，內含 **30 幀 × 25 關節（BODY_25）**。可直接點開瀏覽。

## 結構
```
3DGroundTruth/SceneX/userN/{action}_{L-O-R}.json
```

## 格式（每檔自描述）
| 欄位 | 內容 |
|---|---|
| scene/user/action/instance/layout | 來源 |
| joints | BODY_25（25 關節）|
| frame_coords | **person-centric**：x=左右(L→R 髖)、y=前後、z=離地高度；公尺 |
| fields | `[x, y, z, conf]` |
| keypoints3d | `[30 幀][25 關節][4]` |

> person-centric 表示法：丟掉房間位置、保留左右與離地高度 → 跨場景一致、適合動作 captioning（跌倒=身體貼地、左右肢體語意都在）。

## 各 Scene 狀態
| Scene | instances | 狀態 |
|---|---|---|
| Scene1 | 5003 | ✅ 完成 |
| Scene2 | 4425 | ✅ 完成（訓練用）|
| Scene3 A/B/C | 3142 | ✅ 完成（訓練用，A=1093 B=1065 C=984）|
| Scene4 | 1536 | ✅ 完成（**test-only**，論文 §6.6：train 1-3 → test 4/5）|
| Scene5 | 1678 | ✅ 完成（**test-only**）|

> 全 Scene1–5 共 ~15,784 instances。Scene3/4/5 使用穩健多位置校正
> （`selflabel/scripts/calibrate_scene_multi.py`，每 layout 跨 5 個地板位置自校正，
> reproj 1.7–4.6px）；Scene3 calib 修復前的單位置版本只在 location 1 有效。

## 拿來訓練（展開回 preprocess 結構）
```bash
python selflabel/scripts/unpack_gt.py --gt_dir 3DGroundTruth/Scene1 --scene_root PerceptAlign/data/raw/Scene1
# 之後 preprocess 不要加 --apply_scene_transform（GT 已是最終 person-centric frame）
```
整包也可從 Release 下載 tarball。
