# Rerank top-K của MagicLens-base bằng VLM trên FashionIQ dress

VLM: `Qwen/Qwen2-VL-2B-Instruct` (float16), chấm 50 ứng viên đầu của mỗi truy vấn bằng log-odds logit("Yes") − logit("No"). Điểm mới = z(MagicLens) + w · z(VLM) trên k ứng viên đầu. Truy vấn: dev 930 / test 931 (cùng cách chia với fiq_experiments.py). Chọn k và w theo R_avg của dev. Không mô hình nào được huấn luyện trên FashionIQ; nhãn của nửa dev chỉ dùng để chọn k và w.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| MagicLens zero-shot (baseline) | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| Chỉ VLM, rerank top-50 | 34.4 | 8.1 | 19.3 | 24.9 | 42.1 | 33.5 |
| MagicLens + VLM, top-50 (w=3) | 35.5 | 10.8 | 21.7 | 28.5 | 42.1 | 35.3 |

Chênh lệch trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): R@10 +5.2 [+3.0, +7.2], R_avg +2.6 [+1.5, +3.6] điểm. Rerank trong top-k (k ≤ 50) không đổi tập top-50, nên R@50 giữ nguyên, R_avg chỉ đổi theo R@10 và khoảng tin cậy của R_avg đúng bằng một nửa của R@10. Trên test, ảnh đích được đưa vào top-10 ở 77 truy vấn và bị đẩy ra ở 29 truy vấn.

## Theo độ sâu rerank k (w tốt nhất trên dev cho từng k)

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| top-20 (w=1.5); trần R@10 test 30.6 | 34.2 | 10.7 | 21.6 | 27.4 | 42.1 | 34.7 |
| top-30 (w=2); trần R@10 test 35.7 | 35.4 | 10.8 | 21.4 | 28.2 | 42.1 | 35.2 |
| top-50 (w=3); trần R@10 test 42.1 | 35.5 | 10.8 | 21.7 | 28.5 | 42.1 | 35.3 |
| top-20, chỉ VLM | 33.9 | 9.7 | 19.9 | 26.3 | 42.1 | 34.2 |
| top-30, chỉ VLM | 34.4 | 9.3 | 19.4 | 25.9 | 42.1 | 34.0 |
| top-50, chỉ VLM | 34.4 | 8.1 | 19.3 | 24.9 | 42.1 | 33.5 |

## So sánh thêm trên test (R@10, bootstrap như trên)

- Chỉ VLM (top-50) so với baseline: +1.6 [-0.9, +4.2].
- MagicLens + VLM so với chỉ VLM: +3.5 [+2.0, +5.0].
- Kho ảnh chứa cả ảnh truy vấn (giống cách đánh giá gốc). Ảnh này nằm trong top-10 của 31 truy vấn test trước rerank, còn 5 sau rerank. Nếu lọc ảnh truy vấn khỏi danh sách kết quả của cả hai hệ: R@10 23.52 → 28.46, chênh lệch +4.9 [+2.8, +7.0].

## Phân tích trên test

- Khả năng tách ảnh đích khỏi 49 ứng viên còn lại, trên 392 truy vấn có ảnh đích trong top-50 (AUC trung bình theo truy vấn; 0.5 là ngẫu nhiên): MagicLens 0.739, VLM 0.781, MagicLens + VLM 0.812.
- Tương quan Pearson giữa z(MagicLens) và z(VLM) trên mọi cặp (truy vấn, ứng viên) trong top-50 của 931 truy vấn test: 0.15.

## Toàn bộ lưới: R@10 dev / test (chỉ để tham khảo; k và w chỉ được chọn trên dev)

| w | top-20 | top-30 | top-50 |
|---|---|---|---|
| baseline | 20.3 / 23.3 | 20.3 / 23.3 | 20.3 / 23.3 |
| w=0.1 | 21.0 / 23.7 | 21.1 / 23.7 | 21.1 / 23.8 |
| w=0.2 | 21.4 / 24.4 | 21.6 / 24.6 | 21.7 / 24.7 |
| w=0.3 | 21.9 / 24.8 | 22.0 / 25.0 | 22.0 / 25.3 |
| w=0.5 | 22.4 / 25.5 | 22.9 / 26.1 | 23.3 / 26.6 |
| w=0.7 | 22.9 / 26.3 | 24.2 / 27.2 | 24.5 / 27.5 |
| w=1 | 23.5 / 26.6 | 25.3 / 27.7 | 25.8 / 28.2 |
| w=1.5 | 23.8 / 27.4 | 25.2 / 28.6 | 26.2 / 28.9 |
| w=2 | 23.5 / 27.4 | 26.1 / 28.2 | 26.2 / 29.0 |
| w=3 | 23.4 / 27.3 | 25.4 / 28.2 | 26.3 / 28.5 |
| w=5 | 23.4 / 27.1 | 25.5 / 27.9 | 25.8 / 28.0 |
| chỉ VLM | 23.2 / 26.3 | 24.1 / 25.9 | 24.2 / 24.9 |

## Chi phí

Thời gian chấm VLM: 2.3 giờ GPU cho 93050 cặp (89 ms/cặp, gồm cả đọc ảnh, không gồm tải và nạp mô hình).
