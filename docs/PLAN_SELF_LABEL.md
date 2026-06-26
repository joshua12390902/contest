# PerceptAlign 自製 3D 骨架 GT 執行計畫（誠實版）

> 來源:多 agent workflow(研究+對抗驗證+整合)。Claude 補充更正見文末。

## 1. 一句話結論

**值得做,但只能當「相對效果」復現,不能對齊論文絕對數字。** 你可以用 3 支相機影片自製出 repo `preprocess.py` 能直接吃的 25 關節 3D 骨架 GT,跑通整條 train/eval;但 GT 活在**你自己的地板世界座標系**(非作者棋盤格座標系),且作者釋出的 checkpoint 只是 epoch 0 的 smoke test(等於沒訓練),**絕對 MPJPE 永遠對不上 Table 3**。唯一站得住腳的目標:在你自己的 GT 上**從頭重訓**,量「開/關 geometry conditioning(`rel_rx`)對 cross-layout / cross-scene 退化的影響」—— 這才是論文真正主張、也是唯一可復現的東西。

## 2. 整體流程圖

```
3 支影片 (output1/2/3, 1920x1080, 150 frames@30fps)
   │  ★GT 只取 30 幀(每 5 幀抽 1),不是 150 幀！
   │   preprocess 用 min_t//n_kp_frames 對齊 CSI,餵 150 幀會靜默搞爛配對
   ▼
[A] 2D 偵測  rtmlib RTMW(133-kpt wholebody, ONNX, 無 mmcv) → 每幀單人最大 bbox
   ▼
[B] 相機自校正  ← 真正瓶頸
   │   floor-PnP(點地板已知點 solvePnP);views 1&3 為主,view2 近掠射降權
   │   D435 內參要自校,別盲信 factory fx≈1380
   ▼
[C] 三角化 3D  直接 cv2 DLT/SVD(丟掉 EasyMocap,單人空房零增益)
   │   每關節 ≥2 個高信心視角,conf<0.8 過濾,bone-length 一致性 gate
   ▼
[D] 寫 keypoints3d JSON  <inst>/default/smplx/keypoints3d/000000..000029.json
   │   list 長 1, key "keypoints3d", 25 列 [x,y,z,conf], 公尺
   ▼
[E] 量 tx/rx → geometry_config.json  scene_matrix = 4x4 單位矩陣
   │   triangulate 天線「底座」(別點尖端), RX 點 cluster 中心
   ▼
[F] tools/preprocess.py → 從頭 train → eval(toggle rel_rx 做消融)
```

## 3. 每一步具體做法（含驗證階段修正）

### 步驟 0 — Go/No-Go 前置檢查（先做）
從 HF **完整**下載 ≥1 個 instance,grep 找 `intri.yml`/`extri.yml`/真 keypoints。
- 若作者校正/GT 真的有 → 整個自校正不必做,直接用作者座標系,**絕對 MPJPE 也恢復可比**(最划算分支)。
- 若確認全量都沒有 → 才走自製。同時把 `CONTACT_AUTHORS.md` 寄出(成本零,對沖數週標註)。
- (Claude 補充:我們已用 HF API 確認 5 個 scene 的 instance 都只有 `csi/ csi_mat/ output1-3/`,無 default/calibration。此步是便宜的最終確認。)

### 步驟 A — 2D 偵測
```bash
/workspace/.venv-1/bin/pip install rtmlib onnxruntime-gpu opencv-python
```
- 用 **RTMW wholebody(133 kpt,含腳)**,別用 COCO-17(缺 neck/mid-hip/腳)。
- **★只抽 30 幀**(每 5 幀取 1)。真實 instance 是 30 個 JSON、影片 150 幀 → GT 6fps。餵 150 幀會讓 `t_segment = min_t//n_kp_frames` 算錯。
- 空房單人 → 每幀取最大/最置中 bbox,用 person bbox gate 掉天線桿誤偵測。

### 步驟 B — 相機自校正（瓶頸）
**內參**:別盲信 D435 factory(RGB 可能被 crop/scale,fx 不可靠);用 floor-PnP 點順便估/refine,檢查地磚直線有無桶形畸變。
**外參(floor-PnP)**:
1. 世界座標系:原點=膠帶方框靠 TX 內角;+x 沿地板朝遠端 RX(=TX→RX 基線);+y 沿垂直邊;+z 上;**公尺**。整個 scene 所有 instance 共用此 frame。
2. 每視角點 ≥6 個已知地板點(方框 4 角 + X 標記 + 地磚交點),鋪滿 FoV 對抗平面 IPPE 雙解。
3. `solvePnP(SOLVEPNP_IPPE)` 取兩解挑 reprojection 低那組 → 加 ≥1 非共面點 `SOLVEPNP_ITERATIVE` 殺歧義 → `solvePnPRefineLM` 聯合精修。

**驗證者關鍵修正(必採)**:
- **view2(output2)近掠射 / 近正對地板**,平面 PnP ill-conditioned → **降權,主要用 views 1&3 做 2-view 三角化,view2 只當信心檢查**。
- 地板**鏡面反光**(窗戶倒影、鬼影人)會破壞點選與 2D 偵測 → 預期幾乎全手動;校正幀**逐相機挑「地板被遮最少」那幀**(人站方框內會擋角點,別盲取 frame 0)。
- 膠帶方框多視角下非正方、被紙箱切到、角點出框;若 ≥6 共視點湊不齊 → 改 COLMAP/SfM(用窗框/天花板格),再用地磚固定公尺尺度。

### 步驟 C — 三角化 3D（丟掉 EasyMocap）
驗證者三條一致:**丟 EasyMocap**(單人空房零跨視角關聯增益,卻是最高安裝風險)。
- 直接 DLT/SVD:每關節堆各相機 `P=K[R|T]` + 2D 點(conf 加權),3 視角 SVD / 2 視角 `cv2.triangulatePoints`。~50 行。
- 組 **BODY_25 順序**:Neck=mid(LSh,RSh)、MidHip=mid(LHip,RHip)、腳 19-24 取 RTMW 腳點;合成關節 conf=來源 min/product。
- **conf<0.8 過濾 + 時序插補**(複刻作者 recipe)。

### 步驟 D — 寫 keypoints3d JSON
每幀 `[ {"keypoints3d": [[x,y,z,conf] ×25]} ]`,公尺,放 `<inst>/default/smplx/keypoints3d/`,**30 檔 000000–000029**。Schema 已對 `preprocess.py` 確認(取 `kp_data[0]['keypoints3d'][:25]`,第 4 欄 conf)。

### 步驟 E — 量 tx/rx + geometry_config
- **tx/rx 必須三角化**(你是遠端、不在實驗室,不能假設捲尺量)。這是第二次 mini-calibration。
- 點**天線底座**(不點尖端),RX cluster 點**中心**(模型每 RX 只要一點)。底座被框邊切到 → 改點桿落地點。
- 分 TX/RX:**數天線桿**(TX=1 根,每 RX=3 根);Scene3 用 `assets/figures/scene3_layout{A,B,C}.jpg`(綠=TX,青=RX)+ 印製基線(A 405cm / B 311cm / C 187cm)交叉驗證。Scene1/2/4/5 無圖 → 用 CSI 檔名 `1-r1/r2/r3.mat` 固定映射,全 scene 一致。
- `scene_matrix` = **4×4 單位矩陣**(`kp @ I.T` no-op → 減 tx → tx-relative;模型 `rel_rx=rx-tx`,絕對 frame 自動消掉)。Scene3 用多 layout schema。

### 步驟 F — preprocess → 從頭 train → eval
```bash
cd /workspace/perceptalign_repro/PerceptAlign
PYTHONPATH=$(pwd) /workspace/.venv-1/bin/python tools/preprocess.py \
  --scene_root data/raw/Scene1 --out_root data/preprocessed_actions \
  --apply_scene_transform --max_instances 5
# 確認 n_kp_frames=30、無 "missing tx/scene_matrix" warning、產出 .pt
```
- gotcha:`PYTHONPATH=$(pwd)`、config 用絕對路徑、一律 `/workspace/.venv-1/bin/python`。
- **deliverable 重定義**:作者 checkpoint 是 smoke test、`eval.py` MPJPE 是裸 L2 無 Procrustes/root align → 唯一誠實實驗 = **在你的 GT 上從頭訓,toggle `rel_rx`,量 cross-layout/cross-scene 退化差異**。

## 4. 風險 + 自我驗證 gate
**最可能卡關**:① 校正(無棋盤格、view2 近掠射、鏡面地板)② 共視不足(每關節需 ≥2 視角)③ tx/rx 誤差直灌 `rel_rx` ④ 規模幻覺(Scene1 ≈ 數千 instance,全量數週~數月)。

**訓練前必過的品質 gate**:
- Reprojection RMS **< 5–10 px**
- 骨長 std/mean **< ~3%**(沒真值時最便宜的品質計)
- 身高 head_z−foot_z ≈ **1.6–1.8 m**(否則尺度錯)
- 天線 z ≈ 0–0.9 m;TX-RX 基線對得上 Scene3 cm 圖(<~5cm)
- ⚠️ reprojection≈0 ≠ 全局正確(相似變換可能整體錯)→ 骨長+身高 anchor 是不可省的第二道閘

## 5. 範圍建議（絕不全量）
- **Pilot 先做**:Scene1 的 3–5 個 instance,校正一次,跑通 A→F,過品質 gate。**1–2 週驗證方法可行。**
- **最小可發表**:一個 scene 內 cross-layout 對比 + `rel_rx` 開/關消融。幾百個 instance 足矣。
- **不承諾**:全 Scene1-5 GT、對上 Table 3。每次相機移位都要逐 rig 手動校正,無法平行化。

## 6. 自動化分工
| 步驟 | Claude 可自動跑 | 你要人工 |
|---|---|---|
| 0 前置檢查 | ✅ HF 下載 + grep | 決定走作者 frame 還是自製 |
| A 2D 偵測 | ✅ 裝 rtmlib、寫腳本、3×30 幀全跑 | — |
| B 相機校正 | ✅ 半自動角點輔助、solvePnP/refine、算 RMS、view2 降權 | 🔴 定義座標系、點地板角點/X、量地磚 pitch |
| C 三角化 | ✅ cv2 DLT/SVD、BODY_25 組裝、conf 過濾 | — |
| D 寫 JSON | ✅ 全自動 | — |
| E tx/rx | ✅ 三角化、數桿判 TX/RX、寫 config | 🔴 點天線底座、確認 TX/RX 指派 |
| F train/eval | ✅ 全自動、rel_rx 消融、出曲線 | 抽查重投影目視 |
| 品質 gate | ✅ 自動算 RMS/骨長/身高/基線 | 🔴 看 debug 疊圖抽查、決定 go/no-go |

> Debug 疊圖一律存 `<input>/debug/`。

---
## Claude 補充更正
1. workflow 某 agent 稱「repo 附了 placeholder keypoints(線性 ramp)」—— 那是**我稍早 smoke test 手動塞的假檔**,非作者所附。公開資料根本沒 keypoints。
2. keypoints 私有資料集:我測是 401(private),某 verifier 測到 "not found"(可能已刪)。無論哪種,都拿不到 → 自標是現實路徑,但信照寄(成本零)。
