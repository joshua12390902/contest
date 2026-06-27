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
| Scene2 | — | ⏳ 待標（訓練用）|
| Scene3 A/B/C | — | ⏳ 待標（訓練用）|
| Scene4 / Scene5 | — | ⏳ 待標（**test-only**，論文 §6.6：train 1-3 → test 4/5）|

## 拿來訓練（展開回 preprocess 結構）
```bash
python selflabel/scripts/unpack_gt.py --gt_dir 3DGroundTruth/Scene1 --scene_root PerceptAlign/data/raw/Scene1
# 之後 preprocess 不要加 --apply_scene_transform（GT 已是最終 person-centric frame）
```
整包也可從 Release 下載 tarball。
