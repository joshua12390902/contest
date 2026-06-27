# Contest — PerceptAlign 復現(WiFi CSI → 3D 人體姿態)

本專案以論文 **PerceptAlign**（*Breaking Coordinate Overfitting: Geometry-Aware WiFi Sensing for Cross-Layout 3D Pose Estimation*, MobiCom 2026）為雛形做復現，作為比賽基礎。
原始碼/資料：[Trymore-lab/PerceptAlign](https://github.com/Trymore-lab/PerceptAlign) · arXiv:2601.12252

---

## 這個 repo 解決什麼問題

作者公開的資料集（HuggingFace Scene1–5）**只有 WiFi CSI + 3 路相機影片**，
**沒有訓練必需的 3D 骨架標註（GT）、相機校正、geometry_config**（keypoints 在一個私有資料集裡，拿不到）。

→ 本專案的核心貢獻：**用公開的 3 路影片，自己重建 3D 骨架 GT**，直接餵給 PerceptAlign 模型訓練。

### 關鍵方法（不需作者外參、不需深度、不需地板棋盤格）
```
3 路同步相機 → 同一個人的不同角度 = 天然跨視角對應點
  → 用「人體 2D 關節點」對極幾何自校正（cv2.findEssentialMat / recoverPose）
  → 多視角三角化 → 公制 3D 骨架 → 寫成 repo 的 keypoints3d 格式
```
原本最難的「相機自校正」被鏡面地板擋住（自動偵測失敗），改用**人體關節點**當校正靶後完全繞過。

---

## 成果（Scene1 pilot 已驗證）

| 指標 | 結果 |
|---|---|
| 重投影 RMS | **0.38 px**（2-view）/ 2 px（3-view）|
| 骨長變異 | 0–3% |
| 上臂/前臂/大腿/小腿/肩寬 | 0.28 / 0.24 / 0.39 / 0.39 / 0.38 m（全解剖正常）|

自製 3D 骨架投影回影片，完美貼合真人：

![骨架序列](samples/skeleton_sequence_1-1-1.png)
![多 instance](samples/skeletons_montage.png)

**端到端打通**：影片 → 2D 偵測 → 自校正 → 三角化 → BODY_25 GT → preprocess → 模型 forward/loss/backward。

---

## 📦 已產出的 GT(直接下載,不用重標)

**Scene1 自製 3D 骨架 GT** 已放在 Release:
- 下載:[scene1_gt.tar.gz](https://github.com/joshua12390902/contest/releases/download/scene1-gt-v1/scene1_gt.tar.gz)(112MB,**5003 instance**,各 30 幀 BODY_25)
- 表示法:**person-centric**(地板對齊高度 z + 髖線朝向對齊保左右 + 骨盆水平置中)→ 跨場景一致、適合 captioning
- 解壓到資料集結構:
```bash
tar xzf scene1_gt.tar.gz -C PerceptAlign/data/raw/Scene1/
# 然後 preprocess(GT 已是最終 frame -> 不要加 --apply_scene_transform)
cd PerceptAlign && PYTHONPATH=$(pwd) python tools/preprocess.py --scene_root data/raw/Scene1 --out_root data/pp
```

**Scene2 自製 3D 骨架 GT** 已放在 Release:
- 下載:[scene2_gt.tar.gz](https://github.com/314834007-learn/contest/releases/download/scene2-gt-v1/scene2_gt.tar.gz)(101MB,**4425 instance**,各 30 幀 BODY_25;QC 通過率 89.6%)
- 表示法:**person-centric**(同 Scene1,跨場景一致)
- 解壓:`tar xzf scene2_gt.tar.gz -C PerceptAlign/data/raw/Scene2/`,然後 preprocess **不要加** `--apply_scene_transform`(GT 已是最終 frame)

> Scene2–5 隊友標完,各自跑 `normalize_gt.py` 正規化後同樣可合併。

## Repo 結構
```
PerceptAlign/               # 官方程式碼（MIT，已內含，不用另外 clone）
  perceptalign/ tools/ configs/ assets/ paper/ LICENSE
selflabel/
  scripts/
    detect_2d.py             # 步驟 A：rtmlib RTMW 2D 偵測
    calibrate_triangulate.py # 步驟 B+C：人體關節自校正 + 三角化（單 instance）
    calibrate_scene.py       # turnkey：給一個參考 instance → 自動生該場景 calib + geometry
    enum_scene.py            # 列出某場景所有 instance（含重試）
    make_gt_json.py          # 步驟 D：轉 BODY_25 + 寫 keypoints3d JSON
    batch_full.py            # 整場景批次（平行下載 + GPU + 自動 QC，可續跑，吃 SCENE_REPO/SCENE_RAW env）
    batch_process.py         # 小批次     calib_lib.py
  calibs/                    # ★ 每場景/佈局的校正（相機不同 → 各一份）
    calib_scene1.npz  calib_scene2.npz
    calib_scene3_A.npz  calib_scene3_B.npz  calib_scene3_C.npz   # Scene3 三佈局
    calib_scene4.npz  calib_scene5.npz
    geometry_scene*.json     # 對應的 tx/rx（scene_matrix=I，tx/rx 為地板估計可精修）
  scene1_instances.txt  scene4_instances.txt  scene5_instances.txt
  # scene2/3 清單用 enum_scene.py 自己生（見下）
example_gt/1-1-1_keypoints3d/# 一個 instance 的 GT 範例（30 幀 × 25 關節）
samples/                    # 成果證明圖
docs/
  TRAIN_PY_EXPLAINED.md     # train.py 完整流程 ↔ 論文 Method 對照
  PLAN_SELF_LABEL.md        # 自標方法完整計畫
  PILOT_RESULTS.md          # pilot 數字
  SETUP_NOTES.md            # 跑官方 repo 的踩雷筆記
  CONTACT_AUTHORS.md        # 向作者要原始 GT 的信（備案）
```

---

## 環境
```bash
# Python 3.10 + CUDA 12.1。建議用 GPU。
pip install torch torchvision numpy h5py tqdm pyyaml opencv-python \
            rtmlib onnxruntime-gpu scikit-learn requests huggingface_hub
# onnxruntime GPU 需要 cudnn 在 LD_LIBRARY_PATH（nvidia-cudnn-cu12 套件內）
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH
```

## 怎麼跑（三步）
```bash
# 0. 官方程式碼已內含在 PerceptAlign/，直接用。只需設 HF token 下載資料：
export HF_TOKEN=<你的 HuggingFace read token>   # 下載資料用,別 commit！

# 1. 自標 GT：對一批 instance 下載影片→偵測→三角化→寫 keypoints3d
python selflabel/scripts/batch_full.py \
  --instances_file selflabel/scene1_instances.txt \
  --calib selflabel/calib_scene1.npz \
  --log scene1.log --workers 16
#   GT 會寫進 PerceptAlign/data/raw/Scene1/<inst>/default/smplx/keypoints3d/
#   把 geometry/scene1.json 放到 PerceptAlign/data/raw/Scene1/geometry_config.json

# 2. preprocess → .pt
cd PerceptAlign && PYTHONPATH=$(pwd) python tools/preprocess.py \
  --scene_root data/raw/Scene1 --out_root data/pp --apply_scene_transform

# 3. 訓練（cross_subject / cross_location split）
PYTHONPATH=$(pwd) python tools/train.py --config configs/<your>.yaml
```
細節與踩雷見 `docs/SETUP_NOTES.md`、`docs/TRAIN_PY_EXPLAINED.md`。

---

## 多場景分工（Scene1–5,給隊友認領）

**每個場景相機不同 → 各有自己的校正**（已附在 `selflabel/calibs/`,品質 reproj 2–7px 都驗過）。
**同一場景內相機固定 → 大家共用該場景的 calib,座標一致、GT 可直接合併。**

| 場景 | HF repo（`SCENE_REPO`） | 校正 | instance 清單 |
|---|---|---|---|
| Scene1 | `Atomathtang/Scene1` | `calibs/calib_scene1.npz` | `scene1_instances.txt`（5,868）|
| Scene2 | `atomathtang11/Scene2` | `calibs/calib_scene2.npz` | 用 enum_scene.py 生 |
| Scene3-A(user1,2) | `Atomathtang/Scene3` | `calibs/calib_scene3_A.npz` | scene3 中 user1,2 |
| Scene3-B(user3,4) | `Atomathtang/Scene3` | `calibs/calib_scene3_B.npz` | scene3 中 user3,4 |
| Scene3-C(user5,6) | `Atomathtang/Scene3` | `calibs/calib_scene3_C.npz` | scene3 中 user5,6 |
| Scene4 | `atomathtang11/Scene4` | `calibs/calib_scene4.npz` | `scene4_instances.txt`（1,554）|
| Scene5 | `atomathtang11/Scene5` | `calibs/calib_scene5.npz` | `scene5_instances.txt`（1,684）|

### 隊友認領一個場景怎麼跑
```bash
export HF_TOKEN=<自己的 HF read token>
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn,os;print(os.path.dirname(nvidia.cudnn.__file__))")/lib:$LD_LIBRARY_PATH
export SCENE_REPO=atomathtang11/Scene2          # 認領的場景 repo
export SCENE_RAW=PerceptAlign/data/raw/Scene2   # 本地存放路徑

# (Scene2/3 沒附清單) 先自己生清單：
python selflabel/scripts/enum_scene.py --repo $SCENE_REPO --out scene2_instances.txt

# 切片給多人 + 跑批次標註（沿用該場景 calib）
python selflabel/scripts/batch_full.py \
  --instances_file scene2_instances.txt \
  --calib selflabel/calibs/calib_scene2.npz \
  --log scene2.log --workers 12
```

### Scene3 三佈局要分開跑（相機/天線每佈局不同）
```bash
export SCENE_REPO=Atomathtang/Scene3 SCENE_RAW=PerceptAlign/data/raw/Scene3
python selflabel/scripts/enum_scene.py --repo $SCENE_REPO --out scene3.txt
grep -E '^user[12]/' scene3.txt > s3A.txt   # A=user1,2
grep -E '^user[34]/' scene3.txt > s3B.txt   # B=user3,4
grep -E '^user[56]/' scene3.txt > s3C.txt   # C=user5,6
python selflabel/scripts/batch_full.py --instances_file s3A.txt --calib selflabel/calibs/calib_scene3_A.npz --log s3A.log
python selflabel/scripts/batch_full.py --instances_file s3B.txt --calib selflabel/calibs/calib_scene3_B.npz --log s3B.log
python selflabel/scripts/batch_full.py --instances_file s3C.txt --calib selflabel/calibs/calib_scene3_C.npz --log s3C.log
```

### 想換更準的參考校正？
某場景 calib 若覺得不夠好（reproj 偏大），用 `calibrate_scene.py` 換個動作大的 instance 重生：
```bash
python selflabel/scripts/calibrate_scene.py --repo $SCENE_REPO --raw $SCENE_RAW \
  --ref user1/action11/1-1-1 --out_calib calibs/calib_scene2.npz --out_geom calibs/geometry_scene2.json
```

---

## 已知限制（誠實）
- GT 在**我們自定義的世界座標系**（非作者棋盤格座標系）→ **絕對 MPJPE 不可直接對比論文 Table 3**。
- 能復現的是**方法的相對效果**：開/關 geometry conditioning（`rel_rx`）對 cross-layout / cross-subject 退化的影響 —— 這正是論文「Breaking Coordinate Overfitting」的核心主張。
- 自標 GT 在快速動作/自遮擋的肢體會較噪（已用 conf 過濾 + reproj/骨長自動 QC 把關）。

## 引用 / 致謝
本專案為學術復現，資料與模型架構來自原作者，請引用原論文：
> Songming Jia, Yan Lu, Bin Liu, Xiang Zhang, et al. *Breaking Coordinate Overfitting: Geometry-Aware WiFi Sensing for Cross-Layout 3D Pose Estimation.* MobiCom 2026. arXiv:2601.12252.

## ⚠️ 安全
**絕對不要把 HuggingFace / GitHub token commit 進 repo**。用環境變數傳。
