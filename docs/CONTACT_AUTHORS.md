# 聯絡作者 — 請求 PerceptAlign keypoints 標註與 geometry_config

## 收件對象
- **主收件人**:黃繼陽 Jinyang Huang(共同作者,合肥工業大學副教授)— `hjy@hfut.edu.cn`(本人首頁驗證)
- **CC**:劉斌 Bin Liu(USTC)、張翔 Xiang Zhang(天津大學)— 通訊作者,信箱請至各校系頁面查
- 第一作者 = Songming Jia(= HuggingFace 帳號 `Songming0612`,keypoints 上傳者)
- 公開 fallback 管道:在 https://github.com/Trymore-lab/PerceptAlign 開 Issue

## 寄之前要改:`[你的名字]`、`[你的 email]`;HF 帳號 = `penyi`(LEE Penyi)

---

## 中文版(主要,寄陸方可自行轉簡體)

**主旨**:關於 PerceptAlign (MobiCom 2026) 資料集存取請求

黃繼陽老師 您好:(並 cc 劉斌、張翔老師)

我是國立陽明交通大學(National Yang Ming Chiao Tung University, NYCU)PAIRLab 的碩士生 **[你的名字]**,研究方向為 Wi-Fi CSI 人體感知。近期拜讀並嘗試復現貴團隊 MobiCom 2026 的論文《Breaking Coordinate Overfitting: Geometry-Aware WiFi Sensing for Cross-Layout 3D Pose Estimation》(PerceptAlign),在此向您請教資料集存取的問題。

我已 clone GitHub 上的 `Trymore-lab/PerceptAlign`,並在本地(RTX 3090)完整驗證程式碼可正常運作 —— 包含模型 forward/backward、`tools/train.py` 訓練流程、`tools/eval.py`,以及對真實 Scene1 `.mat` 檔執行 `preprocess.py` 的 CSI 特徵抽取,輸出維度皆正確。

不過從 HuggingFace 下載 Scene1–5 後,我發現公開的資料集僅包含 CSI(.mat)與相機影片,缺少訓練與評估所必需的兩項:

1. **3D keypoints 標註** —— `Songming0612/PerceptAlign_keypoints` 目前為私有(private),無法存取;
2. **各場景的 `geometry_config.json`**(tx/rx 座標與 scene_matrix,目前僅 Scene2 的值可由 repo 取得)。

由於沒有標註,`preprocess.py` 會對每筆 instance 回傳 None,因此無法產生訓練資料。想請問是否方便:

- 將 keypoints 資料集**開放存取權**給我的 HuggingFace 帳號 **`penyi`**,或提供下載連結;
- 一併提供各場景的 **geometry_config**。

這些資料僅用於學術研究與方法復現,後續成果一定會妥善引用貴論文。非常感謝您撥冗,期待您的回覆!

順頌 研安

[你的名字]
國立陽明交通大學 PAIRLab　碩士生
[你的 email]

---

## English 版(備用)

**Subject**: Request for keypoint labels & geometry_config to reproduce PerceptAlign (MobiCom 2026)

Dear Prof. Huang (cc: Prof. Liu, Prof. Zhang),

I'm a master's student at National Yang Ming Chiao Tung University (NYCU), working on Wi-Fi CSI human sensing. I've been reproducing your MobiCom 2026 paper *Breaking Coordinate Overfitting: Geometry-Aware WiFi Sensing for Cross-Layout 3D Pose Estimation* (PerceptAlign).

I cloned `Trymore-lab/PerceptAlign` and verified the full code path runs locally (RTX 3090): the model forward/backward, `tools/train.py`, `tools/eval.py`, and `preprocess.py` feature extraction on real Scene1 `.mat` files — all output the correct shapes.

However, the public HuggingFace scenes (Scene1–5) contain only CSI (.mat) and camera videos. To train/evaluate I'm missing two things:

1. **3D keypoint labels** — `Songming0612/PerceptAlign_keypoints` is currently **private**.
2. **Per-scene `geometry_config.json`** (tx/rx coords + scene_matrix) — only Scene2's values are available in the repo.

Without the labels, `preprocess.py` returns `None` for every instance. Could you please grant my HuggingFace account **`penyi`** access to the keypoints dataset (or share a link), and provide the per-scene geometry configs? For academic research and reproduction only, with proper citation of your work. Thank you very much for your time!

Best regards,
[Your name]
MSc student, PAIRLab, National Yang Ming Chiao Tung University (NYCU)
[your email]
