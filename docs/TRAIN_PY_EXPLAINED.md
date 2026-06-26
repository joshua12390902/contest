# PerceptAlign `train.py` 完整流程解析(對照論文 Method)

對照檔案:`tools/train.py`、`perceptalign/models/perceptalign.py`、`perceptalign/config.py`
對照論文:Figure 3(System Overview)、Figure 4(網路架構)、Eq (20)–(26)

---

## 0. 一句話對應

論文 Figure 3/4 的資料流：
```
CSI → 前處理(去噪/分段/DFS) → CNN(ResNet34) → 特徵 f
                                                    ⊕ 空間嵌入 e_n(tx/rx 幾何, NeRF 編碼)
                                                    ⊕ 時間嵌入 r_t(LTE)
                                                    ⊕ 接收器偏置 s_n
                          → Token u_{n,t} → Self-Attention Transformer → Pose Head → 3D 骨架
監督 = EasyMocap 視覺 3D(我們的版本 = 自製三角化 3D)
```
`train.py` 就是把「已 preprocess 的 .pt(CSI+GT+tx/rx)」餵進這個網路、算 loss、更新權重的迴圈。
**注意:CNN 之前的 CSI 前處理在 `preprocess.py`,不在 train.py。** train.py 拿到的是現成的 CSI 特徵張量。

---

## 1. `train.py` 主流程(由上而下)

### (1) 設定與環境 — `main()` 開頭
- 讀 `--config xxx.yaml`、`_seed_all(seed)` 固定隨機性
- DDP:`WORLD_SIZE>1` 走多卡(`nccl`),否則單卡
- 解析路徑:`preprocessed_root`(.pt 根)、`manifest`、`weights_dir`

### (2) 資料切分 — `_select_by_protocol()` + `_split_train_val()`
讀 `manifest.json`(每筆:scene/user/action/instance/layout/pt_relpath),依 `protocol.type` 切：
| protocol | 切法(對應論文 Domain Splits §5.4) |
|---|---|
| `cross_subject` | 某 user 當 test,其餘訓練(leave-one-subject) |
| `cross_layout` | 某 layout 當 test(Scene3 A/B/C) |
| `cross_scene` | 某 scene 當 test |
→ 再從 train_pool 切出 `val_ratio` 當驗證集。**這對應論文「為什麼能測跨域泛化」**。

### (3) Dataset / DataLoader — `ManifestPTDataset` + `collate_sequences`
- `__getitem__`:`torch.load(.pt)` → 取 `csi_data [T,Nr,3,H,W]`、`keypoints [T,K,3]`、`keypoints_conf`、`tx_coords`、`rx_coords`、`rx_mask`;超過 `max_seq_len` 截斷
- `collate_sequences`:把不等長序列 **pad** 成 batch,產生 `mask`(哪些幀有效),堆疊 tx/rx

### (4) 建模型 — `posenet(...)`(見第 2 節)
```python
model = posenet(num_keypoints=25, rel_rx_coords=zeros(3,3),
                num_layers=4, num_heads=8, pos_enc_depth=10, max_seq_len=120)
```

### (5) 損失 / 優化器 / 排程
| 元件 | 程式碼 | 論文 | ⚠️ 差異 |
|---|---|---|---|
| 損失 | `nn.SmoothL1Loss` | Eq(26) **MSE** | code 用 SmoothL1(Huber),非 MSE |
| 優化器 | `AdamW` | **Adam** | 多了 weight decay |
| 排程 | `ReduceLROnPlateau`(patience10, ×0.5) | **cosine** annealing | 機制不同 |
| AMP | `GradScaler` 混合精度 | — | 加速用 |

### (6) 訓練迴圈(每 epoch)
對每個 batch:
1. 搬上 GPU;由 `conf>0` 算 `kp_valid`(哪些關節有效)
2. **`rel_rx = rx - tx`** ← 這就是 geometry conditioning 的條件輸入(把幾何變成 tx-relative)
3. `pred = model(csi, mask, rel_rx_coords=rel_rx, rx_mask)` → `[B,T,K,3]`
4. `loss = SmoothL1(pred, gt) * valid` 只在有效關節算,`/grad_accum`
5. `scaler.backward()`;每 `grad_accum` 步 `optimizer.step()`(梯度累積 = 模擬大 batch)

### (7) 驗證 + 存檔 — `_eval_loop()` / `_compute_metrics()`
- 算 **MPJPE**(平均關節誤差)、**PCK@20 / PCK@50**(<20/50mm 比例)= 論文評估指標 §6
- `scheduler.step(mpjpe)`;MPJPE 創新低就存 `{exp}_best.pth`
- **這就是你看到的訓練 log 每一行的來源。**

---

## 2. 模型 `posenet` ↔ 論文 Figure 4 + 公式

輸入 `csi [B, T, Nr=3, C=3, 224, 224]`(T 幀、3 接收器、3 通道[相位/振幅/DFS])

### (A) CNN 特徵 `f_{n,t}` — 對應 **Eq (20)**
```
ResNet34 的 conv1→layer1→layer2→layer3(BatchNorm 全換成 GroupNorm,利於小 batch)
→ 全域平均池化 → [B,T,Nr,256] → Linear(256→512) = W_f·f_{n,t}
```
- 論文:`f_{n,t} = Pool(E_θ(X_{n,t})) ∈ R^D`
- 程式碼 `_apply_cnn` + `feature_proj`;為省記憶體**逐 8 幀分塊**跑 CNN

### (B) 空間嵌入 `e_n` — 對應 **Eq (21)(22)**(geometry conditioning 核心)
```
rel_rx(接收器相對 tx 的 3D 座標) → NeRFPositionalEncoding(depth=10)
   Φ(p) = [sin(2^k·π·p), cos(2^k·π·p)]_{k=0..9} , 再接原始 p   ← Eq(21)
→ spatial_mlp(Linear→ReLU→Linear) = g_ψ → e_n ∈ R^512          ← Eq(22)
```
- **這是論文最關鍵的設計**:把 WiFi 收發器幾何「升維編碼」當條件,而不是讓模型死背
- `NeRFPositionalEncoding` 類別 + `spatial_mlp`

### (C) 時間嵌入 `r_t` + 接收器偏置 `s_n` — 對應 **Eq (24)** 後半
- `temporal_embedding`:可學參數 `[1, max_seq_len, 512]`(LTE,跨接收器共享)
- `receiver_bias`:可學 `[1,1,Nr,512]`(每接收器硬體個性)

### (D) Token 構建 — 對應 **Eq (24)**
```python
u = LayerNorm(W_f·f_{n,t} + W_e·e_n + r_t + s_n)   # [B,T,Nr,512]
```
程式碼:`u = f_proj + e_n + r_t + s_n; u = token_norm(u)`
→ 攤平成 `[B, T*Nr, 512]` 序列(= Eq(25) 的 U^{(0)},長度 Nr×T)

### (E) Transformer 編碼 — 對應 **Eq (25)** / Figure 4 中段
```
num_layers 個 TransformerBlock(pre-norm):MultiheadAttention(8 heads) + MLP(ratio 4, GELU)
key_padding_mask 由 mask(有效幀) & rx_mask(有效接收器) 組成 → 忽略 padding
```
⚠️ `num_layers=4`(config 預設),**論文 L=6**(參數量 23.4M vs 29.7M 的差異來源)

### (F) Pose Head 解碼 — 對應 **Eq (25)** `ŷ_t = h_φ(z_t)`
```
把 Nr 個接收器 token 串接 → [B,T,Nr*512] → MLP(Nr*512→1024→512→K*3) → [B,T,25,3]
```
程式碼:`decoder`(3 層 Linear + ReLU)

---

## 3. 完整對照表(論文 ↔ 程式碼)

| 論文元件 / 公式 | 程式碼位置 | 我們復現狀態 |
|---|---|---|
| CSI 前處理(去噪/CSI-ratio/DFS) | `tools/preprocess.py` | ✅ 真資料驗證過 |
| Eq(20) CNN 特徵 ResNet34 | `posenet._apply_cnn`+`feature_proj` | ✅ |
| Eq(21) NeRF 位置編碼 | `NeRFPositionalEncoding` | ✅ |
| Eq(22) 空間嵌入 g_ψ | `spatial_mlp` | ✅ |
| Eq(24) Token = LN(W_f·f+W_e·e+r_t+s_n) | `forward` 中段 | ✅ |
| Eq(25) Transformer + Pose head | `transformer_blocks`+`decoder` | ✅(層數 4 vs 論文 6) |
| Eq(26) 損失 | `train.py` SmoothL1 | ⚠️ 論文寫 MSE |
| 監督訊號 = EasyMocap 3D | 我們 = 自製三角化 3D | ⚠️ 自製座標系 |
| Domain splits §5.4 | `_select_by_protocol` | ✅ |
| 評估 MPJPE/PCK §6 | `_compute_metrics` | ✅ |

**要忠實對齊論文需改**:`num_layers 4→6`、`SmoothL1→MSE`、`AdamW+Plateau→Adam+cosine`。
**本質限制不變**:自製 GT 在自定義座標系 → 絕對 MPJPE 不可比 Table 3,只能復現相對趨勢。
