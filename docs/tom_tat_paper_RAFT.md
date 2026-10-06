# Tóm tắt paper RAFT và cách Table 3 (hàng STSB) được tạo ra

**Paper:** Michal Lukasik, Zhao Meng, Harikrishna Narasimhan, Yin-Wen Chang, Aditya Krishna Menon, Felix X. Yu, Sanjiv Kumar. *Better autoregressive regression with LLMs via regression-aware fine-tuning.* ICLR 2025 (Google Research).

**Quy ước trích dẫn:** "p.N" là số trang của bản PDF proceedings 31 trang (trang 1–10 là bài chính, 11–16 tài liệu tham khảo, 17–31 phụ lục). "§x.y" là mục của paper. Các công thức dưới đây chép lại từ paper (giữ ký hiệu gốc); phần nào là nhận xét/suy luận của người viết tài liệu này thì ghi rõ **[nhận xét]**. Tài liệu không thêm công thức hay con số nào không có trong paper hoặc trong nguồn bên ngoài được dẫn.

---

## 1. Bài toán (§1, §2.1, p.1–3)

* Từ vựng hữu hạn $\mathcal{V}$; tập input $\mathcal{X} \subset \mathcal{V}^*$ (chuỗi token); tập đích thực $\mathcal{Y} \subset \mathbb{R}$, chỉ xét số có biểu diễn thập phân hữu hạn. Mỗi $y \in \mathcal{Y}$ có **một** biểu diễn chuỗi duy nhất $\mathtt{str}(y) \in \mathcal{V}^*$ (vd số nguyên 1 ↦ `"1"`). Phân phối thật $\mathbb{P}(x, y) = \mathbb{P}(x)\cdot\mathbb{P}(y\mid x)$ (p.2–3).
* *Natural language regression*: học $\hat{y}\colon \mathcal{X}\to\mathbb{R}$ cực tiểu sai số bình phương (p.3, công thức không đánh số):

$$L(\hat{y}) = \mathbb{E}_{(x,y^*)\sim\mathbb{P}}\big[(y^* - \hat{y}(x))^2\big].$$

* Bộ dự đoán Bayes-optimal: $\hat{y}(x) = \mathbb{E}_{y^*\sim\mathbb{P}(\cdot\mid x)}[y^*]$ (p.3).
* LLM cho phân phối $p(\cdot\mid x)$ trên chuỗi; có thể $p(z\mid x) > 0$ với $z$ không phải số (p.3).
* Ví dụ ứng dụng: STSB (semantic similarity), đánh giá chất lượng dịch, phân tích cảm xúc (p.1).

Hai họ cách tiếp cận cũ (p.1): **autoregressive regression** (sinh số dưới dạng văn bản, fine-tune bằng cross-entropy) và **predictive head** (học một đầu hồi quy trên biểu diễn của LLM, vd mean-pooling embedding đầu ra, hoặc *logit ở vị trí cuối của một token đặc biệt*, vd `<extra_id_0>` của T5, Fernandes et al. 2023). Nhược điểm nêu ở p.1–2: autoregressive không tận dụng tính "số" của đích (với đích 1, đoán 11 tệ hơn đoán 1.1); predictive head lệch khỏi mục tiêu tiền huấn luyện next-token prediction.

## 2. Các phương pháp nền (§2.2–§2.5)

### 2.1 Suy luận chuẩn (§2.2, p.3)

$$\hat{y}_{\mathrm{AR}}(x) \doteq \mathtt{float}(z), \qquad z \sim p(\cdot\mid x). \tag{1}$$

$z$ sinh tự hồi quy từng token; $\mathtt{float}(z)$ chuyển chuỗi (vd `"12.34"`) thành số; nếu $z$ không biểu diễn số (vd `"banana"`) thì trả một giá trị mặc định — "*Unless otherwise stated, we assume* $\mathtt{float}(z) = 0.0$ *for* $z \notin \mathcal{Y}$" (p.3). Greedy decoding / sampling theo nhiệt độ xấp xỉ mode:

$$\hat{y}_{\mathrm{mode}}(x) := \arg\max_{y\in\mathcal{Y}} p(y\mid x). \tag{2}$$

Có thể dùng constrained decoding, nhưng paper ghi rằng thực tế đầu ra của LLM tốt thường là số kể cả zero-shot (p.3).

### 2.2 RAIL — suy luận nhận biết hồi quy (§2.3, p.3–4; Lukasik et al. 2024)

$$\hat{y}_{\mathrm{RAIL}}(x) = \arg\min_{v\in\mathbb{R}} \mathbb{E}_{y\sim p(\cdot\mid x)}\big[\ell(\mathtt{float}(y), v)\big], \tag{3}$$

với loss bình phương $\ell(y,\hat{y}) = (y-\hat{y})^2$ có nghiệm đóng

$$\hat{y}_{\mathrm{RAIL}}(x) = \mathbb{E}_{y\sim p(\cdot\mid x)}\big[\mathtt{float}(y)\big]. \tag{4}$$

Kỳ vọng trên mọi chuỗi là không tính được chính xác; ước lượng bằng **lấy mẫu** hữu hạn, hoặc **chấm điểm** trên lưới $\mathcal{Y}_{\mathrm{grid}} \subset \mathcal{Y}$ (p.4):

$$\hat{y}_{\mathrm{RAIL}}(x;\mathcal{Y}_{\mathrm{grid}}) = \sum_{y\in\mathcal{Y}_{\mathrm{grid}}} p(\mathtt{str}(y)\mid x)\cdot y. \tag{5}$$

Lưu ý của paper: có thể $\sum_{y\in\mathcal{Y}_{\mathrm{grid}}} p(\mathtt{str}(y)\mid x) \neq 1$ nên (5) không hẳn là kỳ vọng (p.4). Chọn lưới: $\mathcal{Y}$ rời rạc cỡ vừa thì $\mathcal{Y}_{\mathrm{grid}} = \mathcal{Y}$; $\mathcal{Y}$ bị chặn thì lấy các điểm cách đều (số nguyên hoặc số thập phân cố định, vd 2 chữ số) (p.4).

### 2.3 Fine-tuning chuẩn (§2.4, p.4)

$$L(p) = \mathbb{E}_{(x,y^*)\sim\mathbb{P}}\big[\ell(y^*, p(\cdot\mid x))\big] \tag{6}$$

với $\ell\colon \mathcal{Y}\times\Delta_{\mathcal{V}^*}\to\mathbb{R}$; trên mẫu $S \in (\mathcal{X}\times\mathcal{Y})^N$:

$$\hat{L}(p) = \frac{1}{N}\sum_{(x,y^*)\in S}\ell(y^*, p(\cdot\mid x)). \tag{7}$$

Lựa chọn chuẩn là log-loss (cross-entropy):

$$\ell(y^*, p(\cdot\mid x)) = -\log p(\mathtt{str}(y^*)\mid x). \tag{8}$$

Fig. 1(a) (p.2): CE áp lên từng token của biểu diễn chuỗi $y^*_1 y^*_2\cdots y^*_k$. Sau fine-tune có thể dùng standard decoding (§2.2) hoặc RAIL (§2.3) (p.4).

### 2.4 Predictive head (§2.5, p.4–5)

Trích biểu diễn $\Phi(x)\in\mathbb{R}^q$ rồi hồi quy $s\colon\mathbb{R}^q\to\mathbb{R}$, thường tuyến tính $\hat{y}(x) = b + w^\top\Phi(x)$. Ký hiệu (p.4–5): $V = |\mathcal{V}|$, $x$ dài $L$; $\epsilon_{\mathrm{in}}(x) = E_{\mathrm{in}}\epsilon_{\mathrm{oh}}(x)\in\mathbb{R}^{D\times L}$; qua các lớp attention/MLP được $\epsilon_{\mathrm{out}}(x)\in\mathbb{R}^{D\times L}$; logits $f_{\mathrm{out}}(x) = E_{\mathrm{out}}^\top\epsilon_{\mathrm{out}}(x)\in\mathbb{R}^{V\times L}$; $p_{\mathrm{out}}(\cdot\mid x) = \mathrm{softmax}(f_{\mathrm{out}}(x))\in[0,1]^{V\times L}$; với Gemma $E_{\mathrm{in}} = E_{\mathrm{out}}$. Ví dụ $\Phi$: logit ở vị trí cuối của một token đặc biệt $v_*\in\mathcal{V}$ (Fernandes et al. 2023; Zhuang et al. 2023) hoặc mean-pooling embedding đầu ra (Zhuang et al. 2023). Tối ưu trực tiếp MSE; không decoding khi suy luận (p.5).

## 3. Hạn chế lý thuyết (§3, p.5; chứng minh App. A p.18–19)

**Lemma 1** (p.5). Giả sử $|\mathcal{Y}|\ge 2$ và $0\in\mathcal{Y}$, với $N \doteq \max(\mathcal{Y})$. Với mọi $\epsilon\in[0, 0.5]$, tồn tại $\mathbb{P}, p$ sao cho $\mathbb{E}_x\big[\mathbb{D}_{\mathrm{KL}}(\mathbb{P}(\cdot\mid x), p(\cdot\mid x))\big]\le\epsilon$ và

$$\mathbb{E}_x\Big[\big(\mathbb{E}_{y^*\sim\mathbb{P}(\cdot\mid x)}[y^*] - \hat{y}_{\mathrm{mode}}(x)\big)^2\Big] \ge \Big(\frac{N}{2}\Big)^2.$$

Ý nghĩa: CE + standard decoding lệch mục tiêu vì mode có thể xa mean.

**Lemma 2** (p.5). Cùng giả thiết; tồn tại $\mathbb{P}, p$ với $\mathbb{E}_x[\mathbb{D}_{\mathrm{KL}}]\le\epsilon$ và

$$\mathbb{E}_x\Big[\big(\mathbb{E}_{y^*\sim\mathbb{P}(\cdot\mid x)}[y^*] - \hat{y}_{\mathrm{RAIL}}(x)\big)^2\Big] \ge \Big(\frac{N}{4}\Big)^2\epsilon.$$

Ý nghĩa: CE coi mọi dự đoán sai như nhau (đặt khối xác suất vào token "100" hay "1,000" bị phạt như nhau), nên CE + RAIL vẫn có thể sai số lớn (p.5).

## 4. RAFT (§4.1, p.6)

**Definition 1** (p.6) — RAFT loss:

$$\ell_{\mathrm{RAFT}}(y^*, p(\cdot\mid x)) = \big(y^* - \mathbb{E}_{y\sim p(\cdot\mid x)}[\mathtt{float}(y)]\big)^2. \tag{9}$$

Tức là dùng chính predictor RAIL và đo bình phương sai số với đích. Với lưới hữu hạn và tập fine-tune $S$:

$$\hat{L}_{\mathrm{RAFT}}(p;\mathcal{Y}_{\mathrm{grid}}) = \frac{1}{N}\sum_{(x,y^*)\in S}\Big(y^* - \sum_{y\in\mathcal{Y}_{\mathrm{grid}}} p(\mathtt{str}(y)\mid x)\cdot y\Big)^2. \tag{10}$$

Chỉ cần chấm điểm mỗi $y\in\mathcal{Y}_{\mathrm{grid}}$, không cần sampling/decoding khi huấn luyện (p.6).

**Lemma 3** (p.6). Giả sử $\mathcal{Y}\subset\mathbb{R}$, và $\arg\min_{y\in\mathcal{Y}} \in \mathcal{Y}_{\mathrm{grid}}$, $\arg\max_{y\in\mathcal{Y}} \in \mathcal{Y}_{\mathrm{grid}}$. Gọi $p^*(\cdot\mid x)$ là nghiệm cực tiểu RAFT loss (Definition 1) trên mọi phân phối $p(\cdot\mid x)$. Khi đó predictor $\hat{y}_{\mathrm{RAIL}}(x;\mathcal{Y}_{\mathrm{grid}}) = \sum_{y\in\mathcal{Y}_{\mathrm{grid}}} p^*(\mathtt{str}(y)\mid x)\cdot y$ thoả

$$\hat{y}_{\mathrm{RAIL}}(x;\mathcal{Y}_{\mathrm{grid}}) = \mathbb{E}_{y^*\sim\mathbb{P}(\cdot\mid x)}[y^*].$$

Trực giác: mọi số trong $\mathcal{Y}$ là tổ hợp lồi của số nhỏ nhất và lớn nhất (p.6).
**[nhận xét]** Với STSB, $\min\mathcal{Y} = 0 \notin \{1,\dots,5\}$ nên điều kiện của Lemma 3 không thoả theo nghĩa đen; tuy nhiên vì tổng xác suất trên lưới được phép < 1 (Eq. 10 không chuẩn hoá), ŷ = 0 vẫn đạt được bằng cách dồn khối xác suất ra ngoài lưới.

App. C (p.21) so sánh với MBR fine-tuning:

$$\ell_{\mathrm{MBR}}(y^*, p(\cdot\mid x)) = \mathbb{E}_{y\sim p(\cdot\mid x)}\big[(y^* - \mathtt{float}(y))^2\big], \tag{13}$$

$$\hat{L}_{\mathrm{MBR}}(p;\mathcal{Y}_{\mathrm{grid}}) = \frac{1}{N}\sum_{(x,y^*)\in S}\sum_{y\in\mathcal{Y}_{\mathrm{grid}}} p(\mathtt{str}(y)\mid x)\cdot(y^* - y)^2. \tag{14}$$

Lemma 4 (p.21): nghiệm cực tiểu của (14) là phân phối one-hot trên phần tử lưới gần $y^*(x) = \mathbb{E}_{y^*\sim\mathbb{P}(\cdot\mid x)}[y^*]$ nhất ⇒ bị giới hạn bởi độ mịn của lưới; RAFT không gặp vấn đề này. Table 5 (p.22, STSB, Gemma-2 9B): MBR 0.98 vs RAFT 0.40.

## 5. Thống nhất RAFT và predictive head (§4.2, p.6–7)

Với lưới **một-chữ-số** (mỗi phần tử là một token), $p(\mathtt{str}(y)\mid x) = p_{\mathrm{out}}(\cdot\mid x)_{\mathtt{str}(y),L}$, nên (p.6):

$$\hat{y}_{\mathrm{RAIL}}(x;\mathcal{Y}_{\mathrm{grid}}) = \sum_{y\in\mathcal{Y}_{\mathrm{grid}}} y\cdot p(\mathtt{str}(y)\mid x) = \sum_{y\in\mathcal{Y}_{\mathrm{grid}}} y\cdot p_{\mathrm{out}}(\cdot\mid x)_{\mathtt{str}(y),L}.$$

Cả RAFT một-chữ-số và final-token logit có dạng (p.6) $\hat{y}(x) = b + w^\top\Psi\big(f_{\mathrm{out}}(x)_{:,L}\big)$ với $w\in\mathbb{R}^{\mathcal{V}}$. Khác biệt (p.7): **activation** — RAFT: $\Psi$ = softmax; final-token logit: $\Psi$ = đồng nhất. **Vector trọng số** — final-token logit: $w$ one-hot tại $v_*$; RAFT: $w_v = \mathtt{float}(v)$ (vị trí không phải chữ số có trọng số 0). **Khởi tạo** — predictor RAFT lúc khởi tạo trùng RAIL decoding nên đã là một predictor zero-shot mạnh; predictive head khởi đầu với sai số lớn; thực nghiệm RAFT hội tụ nhanh hơn (Fig. 4, Fig. 5).

Các biến thể mới (p.7, dùng trong ablation Table 4):

$$\hat{y}(x) = b + \mathrm{MLP}\big(f_{\mathrm{out}}(x)_{:,L}\big), \tag{11}$$

$$\hat{y}(x) = \sum_{y'\in\mathcal{Y}} w_{y'}\cdot p_{\mathrm{out}}(x)_{\mathtt{str}(y'),L}. \tag{12}$$

App. E.5 (p.26) tổng quát hoá predictor với token tuỳ ý:

$$\hat{y}_{\mathrm{RAFT-NN}}(x) = \sum_{y\in\mathcal{Y}_{\mathrm{grid}}} p(\mathtt{token}(y)\mid x)\cdot y. \tag{15}$$

## 6. Phân loại phương pháp (Table 1 p.3, Table 2 p.7)

**Table 1 (p.3):**

| Approach | Autoregressive | Fine-tuning | Inference | References |
|---|---|---|---|---|
| Zero-shot decoding | ✓ | None | Standard decoding | Kocmi & Federmann (2023) |
| Fine-tuning and decoding | ✓ | Cross-entropy | Standard decoding | Fernandes et al. (2023) |
| Zero-shot RAIL | ✓ | None | Regression-aware decoding | Lukasik et al. (2024) |
| Fine-tuning and RAIL | ✓ | Cross-entropy | Regression-aware decoding | this work |
| Predictive head | ✗ | Target metric | Point estimate | Fernandes et al. (2023) |
| RAFT | ✓ | Target metric | Regression-aware decoding | this work |

**Table 2 (p.7)** — predictor $\hat{y}(x)$ và loss fine-tune:

| Nhóm | Approach | Predictor $\hat{y}(x)$ | Loss |
|---|---|---|---|
| Autoregressive (prior) | Standard decoding zero-shot | $\arg\max_{y\in\mathcal{Y}} p(y\mid x)$ | N/A |
| | RAIL zero-shot | $\sum_{y'\in\mathcal{Y}} y'\cdot p(y'\mid x)$ | N/A |
| | Standard fine-tuning and standard decoding | $\arg\max_{y\in\mathcal{Y}} p(y\mid x)$ | $-\log p(y^*\mid x)$ |
| Autoregressive (this work) | RAIL standard fine-tuning | $\sum_{y'\in\mathcal{Y}} y'\cdot p(y'\mid x)$ | $-\log p(y^*\mid x)$ |
| RAFT (this work) | General RAFT | $\sum_{y'\in\mathcal{Y}} y'\cdot p(y'\mid x)$ | $(\hat{y}(x)-y^*)^2$ |
| | Single-digit RAFT | $\sum_{y'\in\mathcal{Y}_{\mathrm{digits}}} y'\cdot p_{\mathrm{out}}(x)_{y',L}$ | $(\hat{y}(x)-y^*)^2$ |
| Predictive head (prior) | Final-token logit (Fernandes et al.) | $b + f_{\mathrm{out}}(x)_{v^*,L}$ | $(\hat{y}(x)-y^*)^2$ |
| | Pooled output embeddings (Zhuang et al.) | $b + w^\top\mathrm{pool}(\epsilon_{\mathrm{out}}(x))$ | $(\hat{y}(x)-y^*)^2$ |
| Predictive head (this work) | MLP on the final-token logits | $b + \mathrm{MLP}(E_{\mathrm{out}}^\top\epsilon_{\mathrm{out}}(x)_{:,L})$ | $(\hat{y}(x)-y^*)^2$ |
| | Probability-vector projection | $b + w^\top p_{\mathrm{out}}(x)_{:,L}$ | $(\hat{y}(x)-y^*)^2$ |
| | Learnable regression-aware training | $\sum_{y'\in\mathcal{Y}} w_{y'}\cdot p_{\mathrm{out}}(x)_{y',L}$ | $(\hat{y}(x)-y^*)^2$ |

Chú thích Table 2: $b, w$ học được; $v^*\in\mathcal{V}$ là token cố định; $\mathcal{Y}_{\mathrm{digits}}$ là các chữ số phủ khoảng đích, mặc định `'1'`–`'5'`; General RAFT dùng $\mathcal{Y} = \mathcal{Y}_{\mathrm{grid}}$ (p.7).

## 7. Mỗi cột của Table 3 dùng predictor/loss nào

Table 3 (p.8): RMSE theo dataset × phương pháp × Gemma-2 2B/9B; cột fine-tune là mean ± std qua các lần huấn luyện lại. Hàng STSB:

| Gemma-2 | Zero-shot standard decoding | Zero-shot RAIL | Std FT + std decoding | Std FT + RAIL | Predictive head | RAFT |
|---|---|---|---|---|---|---|
| 2B | 1.10 | 1.05 | 0.59 ± 0.01 | 0.57 ± 0.00 | **0.54 ± 0.00** | **0.54 ± 0.01** |
| 9B | 1.37 | 1.29 | 0.58 ± 0.00 | 0.53 ± 0.00 | 0.52 ± 0.00 | **0.51 ± 0.01** |

| Cột | Fine-tune | Predictor khi đánh giá | Căn cứ |
|---|---|---|---|
| Zero-shot standard decoding | không | greedy decoding ⇒ $\mathtt{float}(z)$ (Eq. 1–2) | §5.1 p.8: "*In zero-shot standard decoding, we use greedy decoding*"; Table 1–2 |
| Zero-shot RAIL | không | Eq. 4 ước lượng bằng trung bình 32 mẫu ở $T=1$ | §5.1 p.8: "*in zero-shot RAIL, the predictor is obtained by sampling 32 targets with temperature T = 1*" |
| Standard fine-tuning standard decoding | CE, Eq. 7–8 | greedy (Eq. 1–2) | Table 1–2; §2.4; §5.2 p.9 |
| Standard fine-tuning RAIL | CE, Eq. 7–8 | $\sum_{y'\in\mathcal{Y}} y'\cdot p(y'\mid x)$ (Eq. 4) — **cách ước lượng không được nêu** | Table 2; §3.2 |
| Predictive head | MSE, $(\hat{y}-y^*)^2$ | $b + f_{\mathrm{out}}(x)_{v^*,L}$ (final-token / special-token logit) | Table 2; **Table 4 p.9**: "Special-token logit" Wireless 0.51 (2B), 0.46 (9B) trùng cột Predictive head Wireless của Table 3 (0.51 ± 0.01, 0.46 ± 0.00), còn "Pooled output embeddings (mean)" là 0.50/0.47 |
| RAFT | Eq. 10 với $\mathcal{Y}_{\mathrm{grid}} = \{1,2,3,4,5\}$ | Eq. 5 trên cùng lưới (single-digit RAFT) | §4.1, §4.2, §5.1 p.8 |

## 8. Thiết lập thí nghiệm cho STSB/Gemma-2

| Thiết lập | Giá trị trong paper | Nguồn |
|---|---|---|
| Dataset | STSB (Cer et al. 2017), cặp câu, điểm tương đồng 0–5 | §5.1 p.8 |
| Prompt | "What is the sentence similarity between the following two sentences measured on a scale of 0 to 5: {Sentence #1}, {Sentence #2}. The similarity measured on a scale of 0 to 5 with 0 being unrelated and 5 being related is equal to" | Table 6 p.23 |
| Khoảng đích | [0, 5] | Table 6 p.23 |
| Chia tập | train 4,887 / validation 863 / test 1,500 | Table 7 p.23 |
| Mô hình | Gemma-2 2B, 9B — "*instruction-tuned model families*" | §5.1 p.8 |
| Số lần huấn luyện | 3 (để tính std) | §5.1 p.8 |
| Tham số cập nhật | toàn bộ | App. D p.23 |
| Dropout | 0.1 | App. D p.23 |
| Batch | 16 | App. D p.23 |
| Số bước | 200K, chọn bước tốt nhất bằng validation | App. D p.23 |
| LR | hằng (constant schedule), chọn trên validation trong {1e-4, 1e-5, 1e-6} | App. D p.23 |
| Optimizer | Adafactor, ε1 = 1e-30, ε2 = 1e-3, decay rate = 0.8 (Adam không tốt hơn) | App. D p.23 |
| Lưới RAFT | $\mathcal{Y}_{\mathrm{grid}} = \{1,2,3,4,5\}$; với STSB $\mathcal{Y}_{\mathrm{grid}}\subset\mathcal{Y}$ | §5.1 p.8 |
| Chỉ số | RMSE | §5.1 p.8 |
| Zero-shot | greedy; RAIL = 32 mẫu, T = 1 | §5.1 p.8 |

(PaLM-2: batch 64, không dropout, 5K bước, báo cáo checkpoint cuối; MovieLens: AdamW, cosine — không dùng cho Table 3 STSB/Gemma-2; App. D p.23.)

**Kết quả STSB liên quan khác:** Table 8 p.24 (STSB-1K: 2B zero-shot 1.10, std FT 0.61 ± 0.01, PH 0.58 ± 0.01, RAFT 0.58 ± 0.00; 9B 1.31 / 0.57 ± 0.01 / 0.57 ± 0.01 / 0.56 ± 0.01); Table 10 p.25 (27B STSB: 1.29 / 0.62 / 0.51 / 0.48); Table 11 p.25 (PaLM-2 STSB); Table 12 p.25 (Gemma-2 2B RAFT: RMSE 0.54, Pearson 93.55, Spearman 93.22; 9B: 0.51, 94.30, 94.18); Tables 15–20 p.27–29 (ablation trên PaLM-2: log-loss với đích scale [0,1], chuẩn hoá lưới, khởi tạo embedding, độ mịn lưới, token khác nhau); Fig. 4 p.29 (số bước tới hội tụ trên STSB theo % dữ liệu: ở 100% khoảng 3,500 cho RAFT và 4,200 cho predictive head, đọc từ hình); Fig. 6–7 p.30–31 (Gemma-2 2B trên STSB: phân phối xác suất trên các chữ số 0–5 trước/sau RAFT, entropy tăng sau fine-tune).

## 9. Những chi tiết paper KHÔNG nêu — lựa chọn trong notebook và lý do

| # | Chi tiết không được nêu | Lựa chọn | Lý do / căn cứ |
|---|---|---|---|
| 1 | Checkpoint base hay `-it` | `google/gemma-2-{2b,9b}-it` (base là tuỳ chọn) | §5.1 viết "instruction-tuned model families". **[nhận xét]** Zero-shot RAIL (32 mẫu, T = 1) chỉ tốt hơn greedy chút ít (1.10 → 1.05) và Fig. 7 cho phân phối chữ số rất nhọn trước fine-tune — phù hợp với mô hình đã instruction-tune, nhưng không chứng minh được. |
| 2 | Có dùng chat template không | Không (`USE_CHAT_TEMPLATE=False`); có cờ | Table 6 cho "Input prompt" dạng hoàn thành câu ("… is equal to"). |
| 3 | Ranh giới prompt / số (dấu cách, BOS) | x = `<bos>` + prompt + `" "`; đích CE = str(y*) + `<eos>` | Tokenizer Gemma-2 tách chữ số và giữ khoảng trắng (Gemma 2 report). Kiểm tra trực tiếp trên `tokenizer_gemma2.model`: `" 3"` → `['▁','3']`, `"equal to "` → `['equal','▁to','▁']`, và tokens(prompt+" ") là tiền tố của tokens(prompt+" "+d) với mọi chữ số d. Nếu x kết thúc ở "to" thì token kế tiếp là `'▁'`, công thức một-chữ-số $p_{\mathrm{out}}(\cdot\mid x)_{\mathtt{str}(y),L}$ (§4.2) gần như bằng 0. Fig. 7 cho thấy trước fine-tune gần như toàn bộ khối xác suất nằm trên chữ số ⇒ thiết lập gốc hẳn đặt chữ số ở vị trí kế tiếp. Notebook in khối xác suất trên chữ số ở bước 0 để kiểm tra. `<eos>` để greedy dừng và để p(str(y)\|x) là xác suất cả chuỗi. |
| 4 | Độ chính xác của str(y*) khi fine-tune CE | Làm tròn 2 chữ số thập phân, dạng ngắn nhất (`5.0`, `3.8`, `4.25`, `2.33`) | Nhãn train STS-B: ~91% có ≤ 1 chữ số thập phân, ~6% có 2, ~2.6% có 3, 14 nhãn (0.24%) là số thập phân tuần hoàn ghi 14 chữ số (vd 3.33333333333333) — đếm trên bản CSV STS benchmark (5,749 cặp train). Bản HF lưu float32 nên phải làm tròn. Sai số do làm tròn 2 chữ số: RMSE ≈ 5e-4. |
| 5 | float(z) cụ thể | Số đầu tiên trên dòng đầu của phần sinh (≤ 8 token mới); không có số hoặc ngoài [0, 5] ⇒ 0.0; ghi lại tỉ lệ | §2.2: float(z) = 0.0 khi z ∉ 𝒴. **[nhận xét]** rất nhạy: E[y²] ≈ 7.8 trên test, 1% dự đoán bị thành 0.0 làm MSE tăng ≈ 0.08 (RMSE 0.55 → ≈ 0.62). |
| 6 | "Std FT + RAIL" dùng sampling hay chấm điểm lưới | 32 mẫu, T = 1 (cùng ước lượng với zero-shot RAIL); lưới 0.0..5.0 + `<eos>` (Eq. 5) là cột phụ tuỳ chọn | Table 2 ghi predictor trên toàn $\mathcal{Y}$ (không phải lưới); §5.1 chỉ mô tả một cách ước lượng Eq. 4 là sampling 32 mẫu. |
| 7 | 𝒴_grid STSB: {1..5} (§5.1) hay {0..5} (Fig. 7 vẽ chữ số 0–5) | {1,2,3,4,5}; {0,…,5} là tuỳ chọn | **[nhận xét]** Khi không chuẩn hoá (Eq. 10), số hạng y = 0 đóng góp 0·p('0') = 0 nên hai lưới cho **cùng** predictor và loss; chỉ khác khi chuẩn hoá. |
| 8 | Chuẩn hoá xác suất trên lưới | Không | Eq. 10 không chuẩn hoá; Table 16 (PaLM-2) thấy không khác biệt đáng kể. **[nhận xét]** với {1..5}, chuẩn hoá ép ŷ ≥ 1 trong khi ~23% nhãn validation GLUE (test của paper) < 1. |
| 9 | Token v* của predictive head; khởi tạo b; LR riêng cho b | v* = `<unused0>`; b = 0; cùng optimizer/LR | Table 4 gọi là "special-token logit"; RankT5 dùng token không dùng `<extra_id_10>`; intro nhắc `<extra_id_0>`. Fig. 5 p.30: predictive head bắt đầu ở RMSE ≈ 4.1 trên Wireless (đích 1–5) — khớp với dự đoán ban đầu ≈ 0 (b = 0). Logit lấy sau soft-capping 30·tanh(z/30) của Gemma-2 (đó là logits thật của mô hình). |
| 10 | Dropout đặt ở đâu | Residual dropout 0.1 trên đầu ra mỗi nhánh attention/MLP (sau post-norm của Gemma-2), trước khi cộng residual; tuỳ chọn attention-dropout | HF Gemma-2 chỉ có `attention_dropout` (mặc định 0), không có dropout residual. Residual dropout là kiểu dropout chuẩn của Transformer (Vaswani et al. 2017). |
| 11 | Cờ "multiply by parameter scale" và clipping của Adafactor | Bật (lr·max(ε2, RMS(θ))), clipping threshold 1.0, không momentum, không weight decay | ε1, ε2, decay rate 0.8 đúng tên và mặc định của `t5x/adafactor.py`, nơi `multiply_by_parameter_scale=True`, `clipping_threshold=1.0`, `beta1=None` là mặc định; ε2 chỉ có tác dụng khi cờ này bật. Tương đương HF: `Adafactor(lr=lr, eps=(1e-30,1e-3), decay_rate=-0.8, clip_threshold=1.0, beta1=None, weight_decay=0.0, scale_parameter=True, relative_step=False, warmup_init=False)` — lưu ý **dấu âm** của `decay_rate` vì HF tính β̂₂ₜ = 1 − t^{decay_rate}. Notebook kiểm tra bản port trùng HF. Nếu sai, LR hiệu dụng khác ~1/RMS(θ) lần — lưới LR 2 bậc độ lớn bù được một phần. |
| 12 | Độ chính xác số của trọng số | fp32 master + autocast bf16 (2B); bf16 + stochastic rounding cho 9B trên 1 GPU 80GB | Với LR 1e-5–1e-6 (và parameter scaling), cập nhật tương đối ~1e-6 nhỏ hơn bước lượng tử bf16 (~2⁻⁸) — làm tròn gần nhất "nuốt" cập nhật; stochastic rounding hoặc Kahan summation khắc phục (Zamirai et al. 2020). |
| 13 | Cách tách 4,887/863 (và tổng 5,750 ≠ 5,749) | Hoán vị GLUE train bằng `random.Random(0)`: 863 đầu = val, 4,886 còn lại = train; lưu chỉ số | GLUE STS-B train có 5,749 cặp, validation 1,500, test 1,379 (nhãn ẩn) ⇒ test của paper = GLUE validation. Giữ đúng kích thước val (dùng để chọn mô hình). |
| 14 | Chọn bước cho hai decoder của mô hình CE | Chọn riêng theo val RMSE của từng decoder, trên cùng các run | Không nêu; mỗi cột là một predictor riêng. |
| 15 | LR chọn chung hay theo seed; ddof của std | Chọn LR có val RMSE trung bình qua seed nhỏ nhất; std mẫu (ddof = 1) | Không nêu. ddof = 0 sẽ cho std nhỏ hơn √(2/3) ≈ 0.82 lần với 3 seed. |
| 16 | Tần suất đánh giá | Mỗi 200 bước (và ở bước 0) | Số bước trong Table 21 (p.29) đều là bội của 200; Fig. 5 có điểm đo ~mỗi 200 bước. |
| 17 | Seed điều khiển gì | Thứ tự dữ liệu + mặt nạ dropout; split cố định | Không nêu. |
| 18 | Tham số sampling ngoài T | Sampling thuần (top-k = 0, top-p = 1), tối đa 8 token mới | §5.1 chỉ nêu T = 1. |
| 19 | Gradient clipping toàn cục | Không | Không nêu; T5X mặc định không có (`global_norm_clip_threshold=None`); Adafactor có update clipping. |

## 10. Không nhất quán trong paper (để tránh so sánh nhầm)

* Table 5 (p.22) ghi RAFT trên STSB với Gemma-2 9B có RMSE **0.40**, trong khi Table 3 (p.8) và Table 12 (p.25) ghi **0.51**. Có thể Table 5 dùng thiết lập khác — paper không giải thích.
* Table 8 (p.24, STSB-1K) ghi zero-shot 9B = **1.31**, Table 3 ghi **1.37**; zero-shot không phụ thuộc kích thước tập train và cùng tập test 1,500 nên lẽ ra phải bằng nhau (2B thì cả hai là 1.10).
* Table 7: 4,887 + 863 = 5,750, nhiều hơn 1 so với 5,749 cặp train của GLUE STS-B.
* §E.5 (p.26) nói zero-shot RAIL "tương ứng với RAFT ở bước 0 vì các predictor tương đương", nhưng §5.1 định nghĩa zero-shot RAIL bằng 32 mẫu (ước lượng Eq. 4 trên toàn không gian), còn RAFT ở bước 0 là Eq. 5 trên lưới {1..5}; hai ước lượng nói chung không bằng nhau. Notebook báo cáo cả hai (`rail_sampling` là cột của Table 3, `rail_grid_digits` là cột phụ và trùng đúng RAFT@bước 0).

## 11. Chi phí tính toán và các sai khác chủ đích

* Theo paper: 200K bước × 3 LR × 3 seed × 3 kiểu fine-tune (CE, head, RAFT) × 2 cỡ mô hình = 54 run × 200K bước. **[ước lượng thô, chưa đo trên GPU]** Gemma-2 2B (~2.6e9 tham số), batch 16 × ~78 token (input x trung bình 73.9 token, tối đa 150 — đo bằng tokenizer Gemma-2 trên STS-B) ≈ 6·2.6e9·1,250 ≈ 2e13 FLOP/bước ⇒ khoảng 0.2–0.5 s/bước trên A100 ⇒ 11–28 giờ/run ⇒ cỡ 300–750 GPU-giờ cho 27 run của 2B, nhiều lần hơn cho 9B.
* Notebook mặc định **MAX_STEPS = 6,000**, eval mỗi 200 bước, vẫn chọn bước tốt nhất theo val — **sai khác so với paper**, dựa trên Fig. 4 (p.29): trên STSB với 100% dữ liệu, RAFT hội tụ quanh ~3,500 bước, predictive head ~4,200 (thanh sai số tới ~4,500). Paper không có đường hội tụ cho CE ⇒ cần xem `history` và tăng `MAX_STEPS` nếu val RMSE của CE còn giảm.
* Tuỳ chọn tiết kiệm (đều là sai khác): early stopping; chọn LR bằng 1 seed; LoRA (`USE_LORA`) cho GPU nhỏ — paper cập nhật toàn bộ tham số.
* Notebook không tự bảo đảm ra đúng số của paper vì các chi tiết ở mục 9 có thể khác thiết lập gốc.

## 12. Vì sao không dùng MATLAB

Thí nghiệm cần fine-tune toàn bộ 2.6B–9B tham số của Gemma-2 trên GPU, truy cập trực tiếp logits từng token để tính Eq. 8/10, Adafactor, sampling có kiểm soát. Checkpoint, tokenizer và cài đặt tham chiếu của Gemma-2 thuộc hệ sinh thái PyTorch (Hugging Face Transformers) / JAX. Bài *Local LLMs with MATLAB* của MathWorks mô tả việc dùng LLM qua API hoặc server cục bộ (Ollama) để **suy luận**; Deep Learning Toolbox có các khối transformer và mô hình như BERT/GPT-2 (*Transformer Models: From Hype to Implementation*), nhưng tôi không tìm thấy cài đặt Gemma-2 chính thức cho MATLAB hỗ trợ full fine-tuning như paper. Vì vậy sản phẩm là một Jupyter notebook PyTorch.

## 13. Nguồn tham khảo

* Paper (ICLR 2025): [proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/ae3bb0adcdce28afb5f7e86a79b6cffa-Paper-Conference.pdf) · [OpenReview](https://openreview.net/forum?id=xGs7Ch3Vyo) · [Google Research](https://research.google/pubs/better-autoregressive-regression-with-llms-via-regression-aware-fine-tuning/)
* Lukasik et al. (2024), *Regression-aware Inference with LLMs* (RAIL), Findings of EMNLP 2024: [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.799/) · [arXiv:2403.04182](https://arxiv.org/abs/2403.04182)
* Fernandes et al. (2023), *The Devil Is in the Errors…*, WMT 2023: [ACL Anthology](https://aclanthology.org/2023.wmt-1.100/) · [arXiv:2308.07286](https://arxiv.org/abs/2308.07286)
* Zhuang et al. (2023), *RankT5* (logit của token không dùng `<extra_id_10>` làm điểm): [arXiv:2210.10634](https://arxiv.org/abs/2210.10634)
* Gemma Team (2024), *Gemma 2: Improving Open Language Models at a Practical Size* (tokenizer SentencePiece 256k, split digits, preserved whitespace, byte-level; logit soft-capping): [arXiv:2408.00118](https://arxiv.org/abs/2408.00118)
* Hugging Face, tài liệu Gemma2 (`final_logit_softcapping=30`, `attn_logit_softcapping=50`): [docs](https://huggingface.co/docs/transformers/en/model_doc/gemma2); blog "Welcome Gemma 2" (khuyến nghị eager attention khi fine-tune): [blog](https://huggingface.co/blog/gemma2)
* Tokenizer Gemma-2 dùng để kiểm tra tokenization: đường dẫn `gs://gemma-data/tokenizers/tokenizer_gemma2.model` trong [google-deepmind/gemma](https://github.com/google-deepmind/gemma) (`gemma/gm/text/_tokenizer.py`)
* Shazeer & Stern (2018), *Adafactor*: [arXiv:1804.04235](https://arxiv.org/abs/1804.04235); mặc định T5X: [t5x/adafactor.py](https://github.com/google-research/t5x/blob/main/t5x/adafactor.py); Hugging Face `Adafactor`: [docs](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
* Zamirai et al. (2020), *Revisiting BFloat16 Training* (stochastic rounding / Kahan summation cho cập nhật trọng số bf16): [arXiv:2010.06192](https://arxiv.org/abs/2010.06192)
* Cer et al. (2017), *SemEval-2017 Task 1: Semantic Textual Similarity* (STS benchmark): [doi:10.18653/v1/S17-2001](http://dx.doi.org/10.18653/v1/S17-2001)
* Wang et al. (2018), *GLUE*: [arXiv:1804.07461](https://arxiv.org/abs/1804.07461); dataset card [nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) (STS-B: train 5,749 / validation 1,500 / test 1,379)
* Bản CSV của STS benchmark (dùng để đếm độ chính xác nhãn): [PhilipMay/stsb-multi-mt](https://github.com/PhilipMay/stsb-multi-mt)
* Vaswani et al. (2017), *Attention Is All You Need* (residual dropout): [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
* MathWorks, *Local LLMs with MATLAB*: [blog](https://blogs.mathworks.com/deep-learning/2024/07/09/local-llms-with-matlab/); *Transformer Models: From Hype to Implementation*: [blog](https://blogs.mathworks.com/deep-learning/2024/10/31/transformer-models-from-hype-to-implementation/)
