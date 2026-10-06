# RAFT — tái lập các hàng STSB của Table 3

Mã nguồn và tài liệu để hiểu và chạy lại các hàng **STSB** của **Table 3** trong paper
*"Better autoregressive regression with LLMs via regression-aware fine-tuning"* (Lukasik et al., ICLR 2025, Google Research)
với Gemma-2 2B / 9B: zero-shot (greedy, RAIL), standard fine-tuning (greedy, RAIL), predictive head và RAFT.

> Repo này **chưa** tái lập được các con số của paper: chúng cần GPU và quyền truy cập Gemma-2 trên Hugging Face.
> Môi trường phát triển chỉ có CPU, nên notebook mới được kiểm tra ở chế độ `SMOKE_TEST` (mô hình tí hon ngẫu nhiên, dữ liệu tổng hợp).

## Cấu trúc

```
docs/tom_tat_paper_RAFT.md        Tóm tắt paper: bài toán, Eq. 1–15, Lemma 1–4, Table 1–2, cột nào của Table 3 dùng
                                  predictor/loss nào, toàn bộ thiết lập cho STSB, các chi tiết paper KHÔNG nêu và
                                  lựa chọn đã dùng, các chỗ không nhất quán trong paper, nguồn tham khảo.
notebooks/raft_stsb_table3.ipynb  Notebook tự chứa (không phụ thuộc file nào khác): dữ liệu → mô hình → 6 cột của
                                  Table 3 → bảng tổng hợp đặt cạnh số của paper. Giải thích bằng tiếng Việt, mỗi khối
                                  code ghi rõ công thức tương ứng của paper.
tests/test_notebook.py            Chạy notebook bằng nbclient: unit check, chạy toàn bộ ở SMOKE_TEST, và (tuỳ chọn)
                                  kiểm tra tách tập trên dữ liệu STS-B thật từ CSV.
requirements.txt                  Thư viện cần thiết.
```

## Chạy trên GPU (để có số thật)

1. Chấp nhận license Gemma-2 trên Hugging Face và tạo access token (`HF_TOKEN`).
2. Mở `notebooks/raft_stsb_table3.ipynb` trên Colab (runtime A100) hoặc máy có GPU. Trên Colab: thêm `HF_TOKEN` vào *Secrets*, bỏ comment ô `%pip install`.
3. Ở **Phần 1 (Cấu hình)**: chọn `MODEL_NAME` (`google/gemma-2-2b-it` mặc định, hoặc `google/gemma-2-9b-it`; bản base là tuỳ chọn), kiểm tra `SEEDS`, `LEARNING_RATES`, `MAX_STEPS`.
4. *Run all*. Kết quả:
   * `raft_stsb_results/<model>/zero_shot.json` — hai cột zero-shot;
   * `raft_stsb_results/<model>/runs/<method>_lr<lr>_seed<seed>.json` — mỗi lần fine-tune: lịch sử val/test mỗi 200 bước, bước tốt nhất theo val, dự đoán test, thời gian/bước;
   * `raft_stsb_results/<model>/table3_stsb.md` — bảng theo bố cục Table 3, đặt cạnh số của paper.
   Bị ngắt giữa chừng thì chạy lại: run nào có JSON rồi sẽ được bỏ qua.

Bộ nhớ: 2B full fine-tune (trọng số fp32 + autocast bf16) cần GPU ~40 GB; 9B trên 1 GPU 80 GB cần
`PARAM_DTYPE="bfloat16"` (có stochastic rounding) và nên bật `GRADIENT_CHECKPOINTING`; `USE_LORA=True` cho GPU nhỏ
là **sai khác** so với paper (paper cập nhật toàn bộ tham số).

Chi phí: paper huấn luyện 200K bước cho mỗi (phương pháp, LR, seed). Notebook mặc định 6,000 bước, đánh giá mỗi 200 bước
(dựa trên Fig. 4 của paper: RAFT/predictive head trên STSB hội tụ trong khoảng 1k–4.5k bước) — đây là sai khác có chủ
đích, xem Phần 14 của notebook và mục 11 của `docs/`.

## Chạy thử trên CPU (không cần mạng)

```bash
pip install -r requirements.txt
RAFT_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebooks/raft_stsb_table3.ipynb --output /tmp/smoke.ipynb
# hoặc
pytest -q tests/                      # ~5 phút: unit check + chạy toàn bộ SMOKE_TEST
RAFT_SKIP_SLOW=1 pytest -q tests/     # chỉ unit check (~30 giây)
```

`SMOKE_TEST` tự bật khi không có GPU. Nó dùng một `Gemma2ForCausalLM` 2 lớp khởi tạo ngẫu nhiên, tokenizer mức ký tự
và dữ liệu giống STSB tổng hợp. Số liệu ra **không có ý nghĩa** so với paper; mục đích là kiểm tra code chạy đúng.
Unit check bên trong notebook so sánh:

* predictor RAFT một-chữ-số với $\sum_y y\cdot\mathrm{softmax}(\text{logits})[\mathrm{token}(y)]$ tính ngây thơ;
* loss RAFT với Eq. 10 tính tay, kể cả một ví dụ đồ chơi;
* trie lưới nhiều token, loss CE, predictive head với cách tính từ logits đầy đủ;
* bản port Adafactor (T5X) với `transformers.optimization.Adafactor`.

Kiểm tra tách tập trên STS-B thật (CSV từ [PhilipMay/stsb-multi-mt](https://github.com/PhilipMay/stsb-multi-mt)):
`STSB_CSV_DIR=/path pytest -q tests/test_notebook.py::test_data_pipeline_real_csv` → train 4,886 / val 863 / test 1,500.

## Vì sao không dùng MATLAB

Thí nghiệm cần full fine-tuning Gemma-2 (2.6B–9B tham số) trên GPU với truy cập trực tiếp logits từng token. Checkpoint,
tokenizer và cài đặt tham chiếu của Gemma-2 nằm trong hệ sinh thái PyTorch/JAX; chưa thấy cài đặt Gemma-2 chính thức
cho MATLAB hỗ trợ việc này, nên repo dùng Jupyter notebook (PyTorch + Hugging Face Transformers).
