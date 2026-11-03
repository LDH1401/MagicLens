# Phần C: rerank top-K của MagicLens bằng VLM (Qwen2-VL) trên Google Colab

Ở Phần A, bộ rerank chỉ trộn thêm điểm CLIP văn bản→ảnh và ảnh→ảnh. Hai điểm này lấy từ chính
backbone CLIP bên trong MagicLens, nên không đưa thêm thông tin mới và không cải thiện được kết quả.
Phần C thay chúng bằng một **mô hình thị giác–ngôn ngữ (VLM)**. VLM nhìn cùng lúc ảnh truy vấn,
câu chỉ dẫn và ảnh ứng viên, rồi trả lời ảnh ứng viên có đúng là ảnh truy vấn sau khi sửa hay không.

VLM không được huấn luyện thêm trên FashionIQ, nên kết quả **vẫn là zero-shot**. Máy cá nhân không có
GPU, nên bước chấm điểm bằng VLM chạy trên **Google Colab**; các bước còn lại chạy trên máy.

## 1. Cách hoạt động

1. **Bước 1: MagicLens (zero-shot, giống baseline).** Lấy 50 ứng viên đầu (top-50) cho mỗi truy vấn,
   với tiền xử lý gốc và câu truy vấn `"{c1} and {c2}"`.
2. **Bước 2: VLM chấm từng cặp.** Mỗi cặp (ảnh truy vấn, ảnh ứng viên) được đưa vào
   `Qwen/Qwen2-VL-2B-Instruct` theo đúng thứ tự sau:

   ```
   [ảnh truy vấn] [ảnh ứng viên]
   The first image shows a reference dress. The second image shows a candidate dress.
   Requested change to the reference dress: "{c1} and {c2}".
   Does the candidate dress look like the reference dress after the requested change? Answer Yes or No.
   ```

   Điểm VLM là `logit("Yes") − logit("No")` của token trả lời đầu tiên, tính bằng float32. Mô hình
   không sinh văn bản; mỗi cặp chỉ cần chạy qua mô hình một lần.
3. **Bước 3: trộn điểm.** Trong k ứng viên đầu, điểm mới là
   `z(điểm MagicLens) + w · z(điểm VLM)`. Hàm `z(·)` chuẩn hóa điểm về trung bình 0, độ lệch chuẩn 1
   trên k ứng viên của từng truy vấn, giống bộ rerank ở Phần A. Ngoài ra có thêm cấu hình
   "chỉ VLM" (chỉ xếp theo điểm VLM).
4. **Bước 4: chọn cấu hình và báo cáo.** Chọn k ∈ {20, 30, 50} và w ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1,
   1.5, 2, 3, 5} theo R_avg trên **dev**, rồi báo cáo kết quả trên **test**, kèm bootstrap ghép cặp so
   với baseline. Nếu trên Colab chỉ chấm top-K với K không nằm trong danh sách đó (ví dụ 25), K cũng được
   thử. Cách chia dev 930 / test 931 trùng với Phần A và Phần B.

Ba điều cần biết khi đọc kết quả:

- **Không thử k = 10:** đổi thứ tự bên trong top-10 không làm thay đổi R@10 hay R@50, nên tiêu chí
  R_avg trên dev không đánh giá được lựa chọn này.
- **R@50 luôn giữ nguyên:** rerank bên trong top-50 không thay đổi tập ảnh nằm trong top-50. Vì vậy
  R_avg chỉ thay đổi theo R@10. Nên xem thêm R@1 và R@5.
- **Trần của rerank:** chỉ những truy vấn có ảnh đích nằm trong top-k mới có thể được cải thiện.
  Trên tập test:

  | Rerank top-k | 20 | 30 | 50 |
  |---|---|---|---|
  | R@10 tối đa (baseline hiện tại: 23.3) | 30.6 | 35.7 | 42.1 |

Câu hỏi gửi cho VLM được viết cố định từ trước, không chỉnh sửa theo kết quả test. Nếu muốn thử câu
hỏi khác, chỉ được so sánh trên dev.

## 2. Quy trình tổng thể

```
 Máy cá nhân (JAX, CPU)                  Google Drive                 Google Colab (PyTorch, GPU)
 ──────────────────────                  ─────────────                ───────────────────────────
 vlm_rerank_prepare.py ── bundle.zip ──▶ MyDrive/magiclens_vlm/ ────▶ vlm_rerank_colab.ipynb
                                                                       (chạy vlm_rerank.py)
 vlm_rerank_eval.py ◀── vlm_scores.jsonl ── MyDrive/magiclens_vlm/ ◀── ghi điểm sau mỗi truy vấn
   └─▶ output/fiq-dress-base_vlm_rerank.md / .json
```

## 3. Các tệp

| Tệp | Chạy ở đâu | Vai trò |
|---|---|---|
| `vlm_rerank_prepare.py` | Máy cá nhân (`.venv`, JAX) | Tính top-50 của MagicLens cho 1861 truy vấn và đóng gói `bundle.zip` |
| `vlm_rerank.py` | Colab (PyTorch + transformers) | Chấm điểm từng cặp bằng VLM; chạy tiếp được sau khi bị ngắt. Tệp này được chép sẵn vào `bundle.zip` |
| `vlm_rerank_colab.ipynb` | Colab | Notebook chạy từng bước, có kiểm tra tự động |
| `vlm_rerank_eval.py` | Máy cá nhân (`.venv`) | Trộn điểm, chọn cấu hình trên dev, báo cáo trên test kèm bootstrap |
| `vlm_rerank/candidates.json` | Do `prepare` tạo | Top-50, điểm MagicLens, hạng của ảnh đích và nhãn dev/test của từng truy vấn |
| `vlm_rerank/bundle.zip` | Do `prepare` tạo | Tệp để tải lên Drive, khoảng 288 MB (đã có trong `.gitignore`) |
| `vlm_rerank/vlm_scores.jsonl` và `.meta.json` | Tải về từ Colab | Điểm VLM và cấu hình đã dùng để chấm. Nên commit vào git (dưới 1 MB) để người khác đánh giá lại được mà không cần GPU |

## 4. Hướng dẫn từng bước

### Bước 1. Chuẩn bị trên máy (khoảng 1–2 phút)

Cần chạy `fiq_experiments.py` trước ít nhất một lần, vì script này dùng lại embedding đã cache trong
`cache/fiq-dress-base/`.

```bash
cd magiclens
.venv/bin/python vlm_rerank_prepare.py --subtask dress
```

Kết quả mong đợi (số baseline phải khớp với Phần A):

```
Baseline dev: R@10 20.3, R@50 44.6, R_avg 32.5; target in top-50: 44.6%
Baseline test: R@10 23.3, R@50 42.1, R_avg 32.7; target in top-50: 42.1%
1861 queries x top-50 = 93050 pairs, 3612 images.
Wrote ./vlm_rerank/candidates.json and ./vlm_rerank/bundle.zip (288 MB).
```

Nếu sau này bạn sửa `vlm_rerank.py`, hãy chạy lại bước này để `bundle.zip` có bản mới.

### Bước 2. Tải lên Google Drive

1. Mở [drive.google.com](https://drive.google.com). Ngay trong **My Drive**, tạo thư mục mới tên
   **`magiclens_vlm`** (New → New folder).
2. Tải `vlm_rerank/bundle.zip` vào thư mục đó. **Không giải nén.**

Nếu muốn dùng tên thư mục khác, phải sửa biến `DRIVE_DIR` ở ô 3 của notebook cho khớp.

### Bước 3. Mở notebook trên Colab

1. Vào [colab.research.google.com](https://colab.research.google.com), chọn **File → Upload notebook**,
   rồi chọn tệp `vlm_rerank_colab.ipynb`. Nếu repo đã có trên GitHub, có thể dùng
   **File → Open notebook → GitHub** thay cho bước này.
2. Chọn **Runtime → Change runtime type → T4 GPU → Save**.

### Bước 4. Chạy thử (ô 1–6)

Chạy lần lượt từng ô bằng **Shift + Enter**:

| Ô | Việc | Kết quả mong đợi |
|---|---|---|
| 1 | Kiểm tra GPU | Một dòng có `Tesla T4` (hoặc GPU khác) |
| 2 | Kết nối Drive | Colab hỏi quyền truy cập: chọn tài khoản chứa `bundle.zip`. Hiện `Mounted at /content/drive` |
| 3 | Cấu hình | `OK, kết quả sẽ ghi vào /content/drive/MyDrive/magiclens_vlm/vlm_scores.jsonl` |
| 4 | Cài thư viện | Dòng cuối hiện `transformers 4.57.6` và `GPU: True`. Pip có thể in một đoạn bắt đầu bằng `ERROR: pip's dependency resolver does not currently take into account...`: đó chỉ là cảnh báo xung đột phiên bản, **bỏ qua được**. Chỉ dừng lại nếu ô báo lỗi Python (`Traceback`) hoặc transformers không phải 4.57.6 |
| 5 | Giải nén | `3612 ảnh; ['candidates.json', 'images', 'vlm_rerank.py']` |
| 6 | Chạy thử 3 truy vấn × 20 ứng viên, hai lần | Lần đầu tải mô hình (khoảng 4.5 GB). In điểm VLM của từng truy vấn, hạng của ảnh đích trước/sau, rồi một kết luận (xem dưới) |

Cách đọc kết quả ô 6:

- Ô này chấm cùng 60 cặp hai lần: một lần mã hóa lại mọi ảnh (cách chuẩn của transformers) và một lần
  dùng lại đặc trưng ảnh đã tính (cách nhanh, mặc định).
- **`OK`:** hai cách cho cùng kết quả, ô 7 sẽ dùng cách nhanh.
- **`CẢNH BÁO`:** cách nhanh bị lỗi hoặc lệch so với cách chuẩn. Ô 6 ghi một tệp đánh dấu
  `use_no_vision_cache` lên Drive, và ô 7 tự dùng cách chuẩn. Cách này chậm hơn nhưng kết quả đúng.
  Không cần làm gì thêm.
- **`LỖI`:** cả cách chuẩn cũng không chạy được. Đọc thông báo của `vlm_rerank.py` ngay phía trên. Thường
  gặp: `Non-finite scores` thì sửa `DTYPE = "float32"` và đổi `OUT` ở ô 3; `CUDA out of memory` thì giảm
  `BATCH_SIZE`. Sau đó chạy lại ô 3 rồi ô 6.
- Chỉ có 3 truy vấn, nên **chưa thể kết luận** VLM tốt hay xấu từ ô này. Ô 6 chỉ dùng để kiểm tra mọi thứ
  chạy được và điểm là số hợp lệ.

### Bước 5. Chạy đầy đủ (ô 7)

- Tổng cộng 1861 truy vấn × 50 ứng viên = 93 050 cặp. Thanh tiến trình hiện tốc độ (`pair/s`) và
  thời gian còn lại.
- **Thời gian đã đo:** lần chạy thật mất 2.3 giờ cho toàn bộ 93 050 cặp (khoảng 4.5 giây mỗi truy vấn),
  chưa tính thời gian tải và nạp mô hình. Thanh tiến trình cho biết thời gian còn lại sau vài phút chạy.
- Điểm được ghi lên Drive ngay sau mỗi truy vấn. **Nếu bị ngắt** (hết phiên, mất mạng, đóng tab, hết hạn
  mức GPU): kết nối lại, chạy lại ô 1–5 rồi ô 7. Không cần chạy lại ô 6.
  - Script bỏ qua các truy vấn đã chấm và chạy tiếp.
  - Ô 7 tự đọc lại cách chạy (nhanh hay dự phòng) đã dùng lúc bắt đầu, nên điểm không bị trộn giữa hai cách.
  - Nếu dòng cuối bị cắt dở do ngắt đúng lúc đang ghi, script tự bỏ dòng đó và chấm lại truy vấn tương ứng.
  - Colab ghi lên Drive có độ trễ, nên vài truy vấn cuối trước khi ngắt có thể bị mất và phải chấm lại.
    Điều này không ảnh hưởng tới kết quả.
- Bản Colab miễn phí giới hạn thời gian dùng GPU, và có thể ngắt khi tab không hoạt động lâu. Hãy giữ tab
  mở và không để máy ngủ. Có thể chia ra chạy trong nhiều ngày; kết quả vẫn được cộng dồn.
- **Không đổi cấu hình giữa các lần chạy tiếp** (MODEL, TOP_K, DTYPE). Script sẽ từ chối nếu cấu hình
  khác với lúc bắt đầu, để không trộn lẫn điểm của hai cấu hình. `BATCH_SIZE` thì đổi thoải mái.
  `DTYPE` được để cố định là `float16`, nên lần sau Colab cấp loại GPU khác (T4, L4…) vẫn chạy tiếp được.

### Bước 6. Kiểm tra và tải kết quả về (ô 8–9)

- Ô 8 in số truy vấn đã chấm. Khi thấy `Đã chấm 1861/1861 truy vấn`, chuyển sang ô 9.
- Ô 9 tải về **hai tệp**: `vlm_scores.jsonl` và `vlm_scores.jsonl.meta.json`. Cũng có thể tải hai tệp
  này trực tiếp từ thư mục `MyDrive/magiclens_vlm/` trên giao diện Google Drive.
- Trình duyệt có thể chặn tệp thứ hai, vì một số trình duyệt không cho trang web tự tải nhiều tệp liên
  tiếp. Khi đó chỉ nhận được `vlm_scores.jsonl`. Cho phép trang tải nhiều tệp (biểu tượng ở thanh địa chỉ)
  rồi chạy lại ô 9, hoặc tải tệp `.meta.json` trực tiếp từ Drive.
- Nếu đã đổi `OUT` (ví dụ sang `vlm_scores_fp32.jsonl` sau lỗi tràn số), hãy tải **đúng tên mới** và tệp
  `.meta.json` đi kèm của nó.
- Chép **cả hai tệp** vào thư mục `magiclens/vlm_rerank/` trên máy.

### Bước 7. Đánh giá trên máy

```bash
.venv/bin/python vlm_rerank_eval.py --scores vlm_rerank/vlm_scores.jsonl
# nếu đã đổi OUT: --scores vlm_rerank/vlm_scores_fp32.jsonl
```

Script in bảng kết quả và ghi ra `output/fiq-dress-base_vlm_rerank.md` và `.json`. Nội dung gồm:

- Bảng chính: baseline, "chỉ VLM" và "MagicLens + VLM" (k, w chọn trên dev), với dev R_avg và test
  R@1/5/10/50/R_avg.
- Chênh lệch trên test so với baseline, kèm khoảng tin cậy 95% (bootstrap ghép cặp 2000 lần).
  Nếu khoảng tin cậy nằm hẳn trên 0 thì mức tăng có ý nghĩa thống kê.
- Bảng theo từng độ sâu k (20, 30, 50), kèm trần R@10 của mỗi k.
- So sánh thêm trên test (R@10): chỉ VLM so với baseline, MagicLens + VLM so với chỉ VLM, và mức tăng khi
  lọc ảnh truy vấn khỏi danh sách kết quả (kho ảnh chứa cả ảnh truy vấn, giống code gốc).
- Phân tích: khả năng tách ảnh đích khỏi các ứng viên khác (AUC) của từng loại điểm, và tương quan giữa
  điểm MagicLens và điểm VLM.
- Bảng R@10 dev / test trên toàn bộ lưới k × w, chỉ để tham khảo; việc chọn k và w chỉ dựa trên dev.
- Tổng thời gian GPU và thời gian trung bình cho mỗi cặp.

Kết quả của lần chạy thật và nhận xét nằm ở mục 10 của [README](../README.md#10-phần-c-rerank-bằng-vlm-qwen2-vl-trên-google-colab).

Muốn xem kết quả sơ bộ khi Colab mới chạy được một phần, thêm `--allow_partial`. Kết quả khi đó được
đánh dấu "Chưa đủ" và không so sánh được với các bảng khác.

## 5. Tham số

`vlm_rerank_prepare.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--model_size` / `--model_path` | `base` / `./models/magic_lens_clip_base.pkl` | MagicLens dùng cho bước 1 |
| `--subtask` | `dress` | `dress`, `shirt` hoặc `toptee` |
| `--data_dir` / `--cache_dir` | `./data/fiq` / `./cache` | Dữ liệu và cache embedding (dùng chung với Phần A) |
| `--out_dir` | `./vlm_rerank` | Nơi ghi `candidates.json` và `bundle.zip` |
| `--top_k` | `50` | Số ứng viên giữ lại cho mỗi truy vấn |
| `--seed` | `0` | Seed chia dev/test, phải trùng với Phần A |

`vlm_rerank.py` (thường gọi qua notebook):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--bundle_dir` | `./bundle` | Thư mục đã giải nén `bundle.zip` |
| `--out` | (bắt buộc) | Tệp JSONL kết quả; nên đặt trên Drive |
| `--model` | `Qwen/Qwen2-VL-2B-Instruct` | VLM; `Qwen/Qwen2.5-VL-3B-Instruct` cũng dùng được nhưng chậm hơn |
| `--top_k` | `0` (= 50) | Số ứng viên chấm cho mỗi truy vấn. Phải từ 20 trở lên thì `vlm_rerank_eval.py` mới đánh giá được |
| `--batch_size` | `10` | Số cặp trong mỗi lượt chạy GPU |
| `--max_pixels` | `200704` (448²) | Giới hạn kích thước ảnh đưa vào VLM. Ảnh FashionIQ khoảng 342×444, nên dùng khoảng 192 token mỗi ảnh |
| `--dtype` | `auto` | `float16` trên T4, `bfloat16` trên L4/A100, `float32` khi chạy trên CPU. Notebook truyền cố định `float16` để chạy tiếp được dù Colab cấp loại GPU khác |
| `--limit` | `0` | Chỉ chấm N truy vấn đầu (để chạy thử) |
| `--no_vision_cache` | tắt | Mã hóa lại ảnh ở mọi cặp (chậm hơn, dùng khi ô 6 báo CẢNH BÁO) |
| `--allow_cpu` | tắt | Cho phép chạy không có GPU (rất chậm, chỉ để kiểm thử) |

`vlm_rerank_eval.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--candidates` | `./vlm_rerank/candidates.json` | Do `prepare` tạo |
| `--scores` | `./vlm_rerank/vlm_scores.jsonl` | Tải về từ Colab (cần cả tệp `.meta.json` đi kèm) |
| `--output` | `./output` | Nơi ghi bảng kết quả |
| `--allow_partial` | tắt | Đánh giá khi mới chấm được một phần |

## 6. Thời gian và tài nguyên

| Bước | Thời gian | Ghi chú |
|---|---|---|
| `prepare` (máy, CPU) | khoảng 1–2 phút (đã đo: 49–126 s), RAM tối đa khoảng 2.9 GB | Dùng embedding đã cache |
| Tải `bundle.zip` lên Drive | tùy mạng | 288 MB, 3612 ảnh |
| Chạy thử ô 6 (Colab) | vài phút | Gồm cả tải mô hình 4.5 GB |
| Chạy đầy đủ ô 7 (Colab GPU, float16) | **đã đo:** 2.3 giờ (8315 s), khoảng 4.5 s/truy vấn, 89 ms/cặp | Không gồm tải và nạp mô hình. Các truy vấn đầu chậm hơn (tới 9.3 s) vì đặc trưng ảnh chưa có trong cache |
| `eval` (máy) | vài giây | |

Ba cách `vlm_rerank.py` giảm thời gian dùng GPU:

- **Dùng lại đặc trưng ảnh.** 93 050 cặp tương ứng với khoảng 186 nghìn lượt đưa ảnh vào VLM, nhưng chỉ có
  3612 ảnh khác nhau, và bộ mã hóa ảnh chiếm phần lớn chi phí của mỗi cặp. Vì vậy mỗi ảnh chỉ được mã hóa
  một lần; đặc trưng được lưu trong RAM (khoảng 1.8 GB ở float16) rồi dùng lại.
- **Chỉ tính đúng hai logit cần dùng.** Lớp đầu ra chỉ được tính ở vị trí cuối cùng, và chỉ lấy hiệu
  "Yes" − "No" bằng float32, thay vì tính 151 936 logit cho mọi vị trí. Nếu tính đầy đủ, lớp đầu ra chiếm
  khoảng 14% số phép tính trên GPU (ước lượng của người rà soát); cách này bỏ được gần hết phần đó.
- **Chạy CPU song song với GPU.** Trong lúc GPU chấm batch hiện tại, CPU đọc và tiền xử lý ảnh của batch
  sau. Nếu chạy nối tiếp, phần này mất khoảng 0.2 giây mỗi batch.

## 7. Xử lý sự cố

| Thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Không thấy .../bundle.zip` (ô 3) | Sai thư mục hoặc chưa tải lên | Kiểm tra tệp nằm ở `MyDrive/magiclens_vlm/bundle.zip`, hoặc sửa `DRIVE_DIR` |
| `NameError: name 'OUT' is not defined` (notebook cũ), `Phiên Colab mới chưa có cấu hình`, `transformers đang là ...` hoặc `Chưa giải nén dữ liệu` | Colab đã khởi động lại (bị ngắt, mở lại notebook, Restart session), nên mất biến, thư viện đã cài và dữ liệu đã giải nén; chỉ dữ liệu trên Drive còn lại | Chạy lại ô 1 → 5, rồi chạy lại ô đang làm. Kết quả đã ghi trên Drive vẫn còn và được chạy tiếp |
| `No GPU found` | Runtime chưa bật GPU | Runtime → Change runtime type → T4 GPU |
| `CUDA out of memory` | Batch quá lớn | Giảm `BATCH_SIZE` (ví dụ 4) rồi chạy lại ô 7 |
| `ERROR: pip's dependency resolver does not currently take into account ...` (ô 4) | Pip hạ transformers về 4.57.6 và báo xung đột với gói khác có sẵn trên Colab | Bỏ qua được nếu dòng cuối hiện `transformers 4.57.6` và `GPU: True` |
| `` `torch_dtype` is deprecated! Use `dtype` instead! `` | Cảnh báo của transformers 4.57 | Vô hại, bỏ qua |
| `Non-finite scores ... re-run with --dtype float32` | float16 bị tràn số với mô hình này | Đặt `DTYPE = "float32"` **và** đổi `OUT` sang tên mới (ví dụ `vlm_scores_fp32.jsonl`) ở ô 3, rồi chạy lại ô 3, 6, 7. Cách này chậm hơn khoảng 3–4 lần. Nhớ tải về và đánh giá đúng tên tệp mới |
| `... was made with other settings (...)` | Cấu hình khác với lúc bắt đầu. Trong ngoặc là mục bị khác: `model`, `top_k`, `dtype`, `max_pixels`, `vision_cache` hoặc `transformers` | Dùng lại cấu hình cũ, hoặc đặt `OUT` mới để bắt đầu lại |
| `Dropping a truncated line ...` | Lần trước bị ngắt đúng lúc đang ghi | Bình thường, không cần làm gì |
| Ô 6 báo `CẢNH BÁO` | Cách nhanh bị lỗi hoặc lệch so với cách chuẩn | Không cần làm gì: ô 7 tự dùng `--no_vision_cache`, kể cả sau khi kết nối lại |
| Ô 6 báo `LỖI: lần chạy chuẩn không hoàn thành` | Script lỗi ngay ở cách chuẩn | Đọc thông báo của `vlm_rerank.py` ngay phía trên (thường là `Non-finite scores` hoặc `CUDA out of memory`, xem các dòng trong bảng này) |
| Ô 9 chỉ tải về `vlm_scores.jsonl`, không có `vlm_scores.jsonl.meta.json` | Trình duyệt chặn trang tự tải tệp thứ hai | Cho phép trang tải nhiều tệp (biểu tượng ở thanh địa chỉ) rồi chạy lại ô 9, hoặc tải tệp `.meta.json` trực tiếp từ `MyDrive/magiclens_vlm/` trên Google Drive |
| `Missing vlm_rerank/vlm_scores.jsonl.meta.json` (eval) | Chưa chép tệp `.meta.json` vào `vlm_rerank/` | Tải tệp này như dòng trên rồi chép vào `vlm_rerank/` |
| `Ignoring a truncated last line` (eval) | Tải tệp về đúng lúc Colab đang ghi | Bình thường; dòng đó bị bỏ qua |
| `top_k ... < 20` (eval) | Trên Colab đặt `TOP_K` dưới 20 | Chấm lại với `TOP_K` từ 20 trở lên (đặt `OUT` mới) |
| Colab ngắt, hết hạn mức GPU | Giới hạn của bản miễn phí | Chờ rồi chạy lại ô 1–5 và ô 7; kết quả cũ vẫn còn trên Drive |
| `Only N/1861 queries are scored` (eval) | Chưa chấm xong | Chạy tiếp trên Colab, hoặc dùng `--allow_partial` để xem sơ bộ |
| `Need scored queries in both halves` (eval) | Dùng `--allow_partial` khi mới chấm quá ít truy vấn | Chấm thêm rồi đánh giá lại |
| `... was scored on another candidates.json` (eval) | Đã chạy lại `prepare` với tham số khác | Dùng đúng `candidates.json` đã đóng gói vào `bundle.zip` lúc chấm |

## 8. Những gì đã kiểm thử và chưa kiểm thử

Đã kiểm thử trên máy (CPU, torch 2.15, transformers 4.57.6):

- **`prepare`:** baseline tái lập đúng số của Phần A (test R@10 23.3, R@50 42.1, R_avg 32.7).
- **`vlm_rerank.py` với các Qwen2-VL tí hon:** các mô hình này có cùng kiến trúc, dùng processor và
  tokenizer thật của Qwen2-VL-2B, nhưng trọng số ngẫu nhiên. Có một bản được chỉnh để điểm thay đổi nếu
  vị trí token hoặc thứ tự ảnh bị sai; bản thường không phát hiện được lỗi loại này. Kết quả:
  - Điểm **giống hệt nhau** giữa cách dùng lại đặc trưng ảnh và cách mã hóa lại, và giữa các `batch_size`
    khác nhau (padding giữa các ảnh khác kích thước được xử lý đúng). Điều này đúng trên transformers 4.49.0,
    4.51.3, 4.52.4, 4.54.1 và 4.57.6, và đúng cả với kiến trúc Qwen2.5-VL.
  - Ở fp32, cách tính hai logit chỉ ở vị trí cuối cho điểm trùng khớp với cách tính đầy đủ. Ở bf16, điểm
    không còn bị làm tròn thành bậc 1/64.
  - Các tình huống đều xử lý đúng: chạy tiếp sau khi ngắt, dòng cuối bị cắt dở, từ chối khi đổi cấu hình
    (kể cả đổi cách chạy nhanh/dự phòng), từ chối khi không có GPU. Kết quả khi chạy tiếp giống hệt một lần
    chạy liền mạch.
- **`vlm_rerank.py` với Qwen2-VL-2B thật:** xem mục "Kiểm thử với mô hình thật" ngay bên dưới.
- **`vlm_rerank_eval.py` với ba bộ điểm giả lập có đáp án biết trước:**
  - điểm "biết đáp án" (oracle) cho R@10 đạt đúng mức trần (30.6 / 35.7 / 42.1);
  - điểm trùng với MagicLens cho ra đúng baseline;
  - điểm ngẫu nhiên không cải thiện được kết quả, và cách chọn trên dev tự chọn trọng số nhỏ nhất.
  Ngoài ra, script đánh giá xử lý đúng các trường hợp: dòng cuối bị cắt dở, dòng hỏng ở giữa tệp, thiếu
  truy vấn thuộc dev hoặc test, và `top_k` dưới 20.
- **Notebook:** đúng chuẩn nbformat. Các ô 3, 5, 6, 7, 8 đã được chạy thật trên máy, với thư mục tạm thay
  cho Drive và mô hình tí hon thay cho GPU, theo ba kịch bản:
  - **Bình thường:** ô 6 báo `OK`. Sau khi "kết nối lại", ô 7 chạy tiếp đúng chỗ.
  - **Cách nhanh bị lỗi:** ô 6 báo `CẢNH BÁO`, ô 7 dùng cách dự phòng, và vẫn giữ cách đó khi chạy tiếp.
  - **Cách chuẩn bị lỗi:** ô 6 dừng với thông báo `LỖI` rõ ràng.
- **Rà soát độc lập:** ba người rà soát kiểm tra riêng phần chấm điểm, phương pháp đánh giá, và notebook
  cùng hướng dẫn. Cả 12 vấn đề họ nêu (đều ở mức nhỏ) đã được sửa và kiểm tra lại.

Kiểm thử với mô hình thật:

- **Kết quả:** `Qwen/Qwen2-VL-2B-Instruct` tải và chạy được. Thử với 1 truy vấn × 3 ứng viên trên CPU,
  dùng bf16, ảnh thu nhỏ (`--max_pixels 50176`) và có dùng lại đặc trưng ảnh.
  - Bản code hiện tại cho điểm là số hữu hạn: `[-0.9567, -1.9521, -0.46]`. Ảnh đích đang đứng hạng 3 theo
    MagicLens được VLM chấm cao nhất, nên lên hạng 1.
  - Bản code trước khi sửa cho cùng thứ tự, nhưng điểm bị làm tròn: `[-1.0, -2.0, -0.375]`.
  - Đây chỉ là một ví dụ, không đủ để kết luận.
- **Hạn chế của máy cá nhân:** máy không đủ RAM trống để chạy float32 (cần khoảng 9 GB). Với bf16, CPU mất
  khoảng 53 giây cho mỗi cặp, nên không thể chạy toàn bộ trên máy; đó là lý do cần Colab.

Chạy thật trên Colab:

- Toàn bộ 1861 truy vấn × 50 ứng viên đã được chấm trong một lần chạy liền mạch, ở float16, có dùng lại
  đặc trưng ảnh (`vision_cache: true` trong tệp meta) và transformers 4.57.6. Cả 93 050 điểm đều là số
  hữu hạn, nằm trong khoảng −7.16 đến +1.39, nên float16 không bị tràn số.
- Kết quả được kiểm tra độc lập theo ba hướng:
  - **Tính lại toàn bộ:** mọi con số trong `output/fiq-dress-base_vlm_rerank.json` (kể cả khoảng tin cậy
    bootstrap, AUC, tương quan và lưới k × w) đã được tính lại bằng code viết riêng, không dùng lại code
    của `vlm_rerank_eval.py`, và khớp hoàn toàn.
  - **Điểm khớp đúng ứng viên:** AUC của điểm VLM khi tách ảnh đích, tính trên cả dev và test (807 truy
    vấn có ảnh đích trong top-50), là 0.763 (riêng test: 0.78, như ở README); nếu xáo trộn điểm giữa các
    ứng viên thì chỉ còn khoảng 0.50. Các ảnh giống hệt nhau từng byte (khác mã sản phẩm) nhận điểm gần
    như bằng nhau (chênh trung bình 0.0014), kể cả khi nằm ở các batch khác nhau. Như vậy điểm không bị
    gán nhầm ứng viên và không phụ thuộc vào cách ghép batch.
  - **Không rò rỉ nhãn:** script chấm điểm không đọc ảnh đích, hạng hay nhãn dev/test. Cách chia dev/test
    và baseline trùng khớp từng truy vấn với Phần A và B. Câu hỏi, lưới k, w và cách chọn đã được viết
    xong trước khi có điểm.
- **Điểm bằng nhau:** ở 11 truy vấn, ảnh đích bằng điểm với một bản sao giống hệt của nó (khác mã sản
  phẩm) trong top-50. Giống cách tính hạng của baseline, trường hợp bằng điểm được tính có lợi cho ảnh
  đích. Nếu tính bất lợi, R@10 test là 28.4 thay vì 28.5, và mức tăng là +5.0 [+2.9, +7.2] thay vì +5.2
  [+3.0, +7.2].

Chưa kiểm thử:

- **Chưa so sánh float16 với float32 trên toàn bộ dữ liệu.** float16 cho điểm hợp lệ, nhưng chưa biết
  float32 cho kết quả khác bao nhiêu.
- **Kết quả ô 6 và loại GPU không được lưu lại.** `vision_cache: true` chỉ cho biết lúc ô 7 bắt đầu, trên
  Drive không có tệp đánh dấu `use_no_vision_cache` (ô 6 không báo `CẢNH BÁO`, hoặc không được chạy). Các
  kiểm tra trên dữ liệu ở trên cho thấy điểm gắn đúng ứng viên và không phụ thuộc batch. Tuy vậy, trên mô
  hình thật chỉ ô 6 so sánh trực tiếp cách dùng lại đặc trưng ảnh với cách chuẩn, mà kết quả ô này không
  được lưu. Nếu chạy lại, nên lưu notebook sau khi chạy (File → Save a copy in Drive).

## 9. Lưu ý khi viết báo cáo

- **Vẫn là zero-shot**, vì không học gì trên FashionIQ; nhãn của nửa dev chỉ dùng để chọn k và w (giống
  Phần A). Tuy vậy, đây là hệ thống **hai mô hình**: Qwen2-VL là một mô hình riêng, được tiền huấn luyện
  trên dữ liệu web lớn mà nhóm tác giả không công bố đầy đủ, nên không loại trừ được việc mô hình đã thấy
  ảnh sản phẩm giống FashionIQ. Cần nói rõ điều này, và không gọi kết quả là "MagicLens" một mình.
- **Chi phí khác hẳn nhau.** Với mỗi truy vấn, MagicLens chỉ cần mã hóa câu chỉ dẫn (khoảng 0.07 giây
  trên CPU, README mục 8.7) rồi xếp hạng cả kho bằng một phép nhân ma trận (vài mili giây). VLM cần chấm
  50 cặp bằng một mô hình 2 tỉ tham số trên GPU (đo được khoảng 4.5 giây mỗi truy vấn). Khi so sánh, nên
  báo cáo cả độ chính xác lẫn thời gian.
- **Lấy R@10 (kèm R@1, R@5) làm kết quả chính.** R@50 không đổi, nên mức tăng R_avg chỉ là mức tăng R@10
  chia đôi, và khoảng tin cậy của nó không phải bằng chứng riêng.
- **Nói rõ kho ảnh chứa cả ảnh truy vấn** (giống code gốc của MagicLens), và VLM đẩy ảnh này xuống. Nên
  báo cáo thêm mức tăng khi lọc ảnh truy vấn khỏi kết quả: R@10 +4.9 [+2.8, +7.0].
- **Không được dò câu hỏi gửi VLM trên test.** Câu hỏi đã được viết cố định từ trước. Mọi thử nghiệm câu hỏi
  khác chỉ được so sánh trên dev.
- **Trích dẫn VLM:** Qwen2-VL-2B-Instruct phát hành theo giấy phép Apache 2.0. Bài báo: P. Wang và cộng sự,
  *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*, 2024
  ([arXiv:2409.12191](https://arxiv.org/abs/2409.12191)).
