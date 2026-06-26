# PerceptAlign 復現筆記

論文:Breaking Coordinate Overfitting: Geometry-Aware WiFi Sensing for Cross-Layout 3D Pose Estimation (arXiv 2601.12252, MobiCom 2026)
Repo:https://github.com/Trymore-lab/PerceptAlign

## 環境(本機 RTX 3090)

- GPU:RTX 3090 24GB,driver 535,CUDA 可用
- Python 環境:`/workspace/.venv-1/bin/python`(torch 2.5.1+cu121, torchvision 0.20.1)
- 額外裝的套件:`h5py`(其餘 numpy/tqdm/pyyaml/torchvision 都已有)

## 已驗證(2026-06-25,用合成資料,不需下載 483GB)

| 步驟 | 結果 |
|---|---|
| 模型 build + forward/backward (GPU) | ✅ 輸出 [B,T,25,3] 正確,梯度有流,峰值 3.2GB |
| `tools/train.py` 全鏈路 | ✅ protocol 切分 / dataset / 訓練 / 驗證 / 存 checkpoint |
| `tools/eval.py` | ✅ 載 checkpoint + test split 評估 |

合成資料在 `PerceptAlign/data/preprocessed_actions/`(Scene1, user1-3, 15 筆,156MB)。
Smoke config:`PerceptAlign/configs/_smoke.yaml`(epochs=2, 絕對路徑, num_workers=0)。
**這些都是假資料,只證明程式碼能跑,MPJPE 數字無意義。**

## 跑這個 repo 的「README 沒講」的雷

1. **要設 `PYTHONPATH`**:repo 沒 setup.py,跑前要 `PYTHONPATH=$(pwd) python tools/train.py ...`
2. **train.py 一定要 `--config`**:README 寫的 `torchrun tools/train.py` 會直接報錯
3. **config 路徑是相對於 config 檔所在目錄**解析的 → 建議 config 內用絕對路徑(或把資料放 configs/ 底下)
4. **`requirements.txt` 不存在**(404),自己裝:torch torchvision numpy h5py tqdm pyyaml
5. **論文 vs code 差異**:config 預設 `num_layers: 4`(→ 23.41M 參數),論文是 6 層(→ 29.71M)。loss 是 SmoothL1(論文寫 MSE),optimizer AdamW+ReduceLROnPlateau(論文寫 Adam+cosine)。要對齊論文數字需手動改。

## 複製這次 smoke test 的指令

```bash
cd /workspace/perceptalign_repro/PerceptAlign
PYTHONPATH=$(pwd) /workspace/.venv-1/bin/python tools/train.py --config configs/_smoke.yaml --device cuda:0
PYTHONPATH=$(pwd) /workspace/.venv-1/bin/python tools/eval.py --config configs/_smoke.yaml \
    --checkpoint weights/smoke_cross_subject_best.pth --split test --device cuda:0 --num_workers 0
```

## 下一步(用真實資料)

1. HF 申請 keypoints 存取:`Songming0612/PerceptAlign_keypoints`(目前 gated 401)
2. 下載一個場景試:`huggingface-cli download Atomathtang/Scene1 --repo-type dataset --local-dir data/raw/Scene1`
3. 確認 `Scene1/` 根目錄有 `geometry_config.json`(座標校正,隨 dataset 釋出)
4. 前處理(先 `--max_instances 5` 驗證):
   `PYTHONPATH=. python tools/preprocess.py --scene_root data/raw/Scene1 --out_root data/preprocessed_actions --apply_scene_transform --max_instances 5`
5. 用真 config 訓練(記得 PYTHONPATH + 絕對路徑)
