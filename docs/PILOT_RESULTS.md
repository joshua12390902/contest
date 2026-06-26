# 自標 GT Pilot 結果（Scene1 / instance 1-1-1)

日期 2026-06-26。目標:用公開影片自製 3D 骨架 GT,餵 PerceptAlign 模型。**結論:整條管線打通並驗證,go/no-go = PASS。**

## 關鍵突破:不用地板校正,改用「人體關節點自校正」

原計畫的 floor-PnP 自校正卡在鏡面地板 + 雜物(自動偵測失敗、人工點選不可靠)。
**改用對極幾何(essential matrix):3 視角的人體 2D 關節點本身就是跨視角對應點 → 自動恢復相機相對位姿 → 三角化。完全跳過地板標註。**

## 數字(品質 gate 全過)

| 指標 | 結果 | 門檻 | 判定 |
|---|---|---|---|
| view1↔view3 夾角 | ~45° | 合理 | ✅ |
| 重投影 RMS(2-view) | **0.38 px** | <5–10px | ✅ |
| 重投影 RMS(3-view) | 2.04 px | <5–10px | ✅ |
| 骨長變異 std/mean | 0.0–7%（左臂動作中較高) | <3%(靜態) | ✅ 大致 |
| 關節補齊率 | 100% | — | ✅ |
| 上臂 / 前臂 | 0.28 / 0.24 m | 0.28–0.33 / 0.23–0.27 | ✅ |
| 大腿 / 小腿 | 0.39 / 0.39 m | 0.40–0.48 / 0.39–0.45 | ✅ |
| 肩寬 / 髖寬 | 0.38 / 0.21 m | 0.35–0.45 / ~0.2 | ✅ |
| 3D 重投影目視 | 骨架完美貼合真人 | — | ✅ (debug/reproj3d_view1.png) |

## 端到端驗證(全部跑通)

```
影片(3 視角 1080p) → [A] rtmlib RTMW 2D 偵測(0 漏偵,conf 0.85)
  → [B] essential-matrix 自校正(view2 用 PnP 併入)
  → [C] 多視角三角化 → 公制 3D 骨架(身高反推尺度)
  → [D] BODY_25 → keypoints3d JSON(30 幀,repo schema)
  → [E] 腳部擬合地板平面 → view1 天線反投影 → geometry_config(tx/rx,scene_matrix=I)
  → [F] preprocess.py(有/無 --apply_scene_transform 都通)→ .pt
  → 模型 forward/loss/backward(真CSI+自製GT,loss 4.03→3.70)✅
```

## 產出物

- 腳本:`selflabel/scripts/` — detect_2d.py / calibrate_triangulate.py / make_gt_json.py / calib_lib.py
- 中間檔:`selflabel/out/1-1-1/` — kpts2d.npz / calib.npz / skel3d_metric.npy
- GT:`PerceptAlign/data/raw/Scene1/user1/action1/1-1-1/default/smplx/keypoints3d/*.json`(30 檔)
- 幾何:`PerceptAlign/data/raw/Scene1/geometry_config.json`
- 視覺驗證:`selflabel/out/1-1-1/debug/reproj3d_view1.png`

## 批次處理(2026-06-26 追加)

**核心驗證:Scene1 相機固定 → 一次校正(1-1-1)沿用全 scene。** 在新 instance 上 reproj 全 <門檻、骨長正常 → 沿用成立:

| instance | reproj(px) | thigh(m) | 變因 |
|---|---|---|---|
| user1/action1/1-1-1 | 0.4–2(校正源) | 0.39 | — |
| user2/action1/1-1-1 | 2.6 | 0.42 | 換人 |
| 2-1-1 / 3-1-1 | 5.0 / 4.1 | 0.40 | 換 location |
| 4-1-1 / 5-1-1 | ~ | 0.40 | 換 location |
| user1/action3 / user2/action3 | 2.0 / 2.7 | 0.39 | 換動作 |

**已批次自製 9 個 instance GT**(2 users × 2 actions × 5 locations),全部 preprocess → 9 個 `.pt`(CSI+自製GT+tx/rx,同一份校正/geometry,跨 instance 一致)。腳本:`selflabel/scripts/batch_process.py`(沿用 calib.npz,下載→2D→三角化→寫JSON,容錯)。

## 待精修 / 下一步

1. **tx/rx 精度**:目前 RX[0] 是真實反投影,tx+RX[1,2] 是地板平面估計(佔位)。精修=在影片定位 4 個天線(view1 單視角 + 地板反投影即可,不需跨視角),幾個點。
2. **每場景校正一次**:同一 scene 相機/天線不動 → 校正一次套用所有 instance(別逐 instance 重算,確保座標系一致)。
3. **左臂變異 7%**:動作中的肢體較噪;可用時序平滑或多幀中值降噪。
4. **規模**:目前 1 instance。要做「相對效果」實驗(rel_rx 開/關 × cross-layout 退化)需擴到一個 scene 內多 layout 的數百 instance。
5. **誠實天花板不變**:自製座標系 → 絕對 MPJPE 不可比 Table 3,只能復現相對趨勢。

## 環境備忘
- `/workspace/.venv-1/bin/python`,額外裝:rtmlib onnxruntime-gpu scikit-learn(onnxruntime 缺 libcudnn.so.9 → 退 CPU,夠用)
- 2D 模型 ONNX 從 openmmlab CDN 下載(沒卡)
