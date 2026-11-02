# MagicLens-base trên FashionIQ dress: cải tiến không cần huấn luyện

Kho ảnh: 3653 ảnh. Truy vấn: 1861 (dev 930 / test 931, chia ngẫu nhiên với seed 0). Mọi lựa chọn dựa trên R_avg = (R@10 + R@50)/2 của dev; các cột test là kết quả báo cáo.

## Tóm tắt (cộng dồn từng cải tiến)

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| Baseline (original, "and") | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| + Tiền xử lý: original | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| + Truy vấn: swap | 33.0 | 7.6 | 18.0 | 22.8 | 42.1 | 32.4 |
| + Rerank top-100 (λ=0.6, μ=0.3) | 33.5 | 7.3 | 17.1 | 23.5 | 43.0 | 33.2 |

Chênh lệch so với baseline trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): R@10 +0.2 [-1.1, +1.6], R_avg +0.5 [-0.5, +1.6] điểm.

## Baseline trên toàn bộ truy vấn

R@10 21.8, R@50 43.4 trên 1861 truy vấn. Paper: R@10 21.5, R@50 41.3 trên tập đầy đủ (không thiếu ảnh).

## 1. Tiền xử lý ảnh (truy vấn "and")

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| original | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| crop | 31.6 | 7.5 | 16.0 | 21.1 | 41.9 | 31.5 |
| pad | 32.3 | 7.1 | 17.6 | 23.1 | 42.4 | 32.8 |

## 2. Cách viết truy vấn (tiền xử lý original)

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| and: "{c1} and {c2}" | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| swap: "{c2} and {c1}" | 33.0 | 7.6 | 18.0 | 22.8 | 42.1 | 32.4 |
| that: "a {cat} that {c1} and {c2}" | 30.2 | 5.6 | 14.3 | 20.8 | 40.9 | 30.9 |
| same: "same {cat} but {c1} and {c2}" | 30.8 | 6.7 | 16.2 | 22.6 | 41.0 | 31.8 |
| ensemble (and, swap, that, same) | 32.6 | 7.2 | 17.4 | 22.2 | 42.7 | 32.5 |

## 3. Rerank top-100

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| Không rerank (original, swap) | 33.0 | 7.6 | 18.0 | 22.8 | 42.1 | 32.4 |
| Chỉ văn bản → ảnh (λ=0.0, μ=0) | 33.0 | 7.6 | 18.0 | 22.8 | 42.1 | 32.4 |
| Chỉ ảnh → ảnh (λ=0, μ=0.0) | 33.0 | 7.6 | 18.0 | 22.8 | 42.1 | 32.4 |
| Cả hai (λ=0.6, μ=0.3) | 33.5 | 7.3 | 17.1 | 23.5 | 43.0 | 33.2 |

Trần của rerank: 51.8% truy vấn test có ảnh đích trong top-100 của bước 1.

## Thời gian (CPU)

Mã hóa ảnh 0.38 s/ảnh, mã hóa văn bản 0.071 s/câu, rerank ≤ 0.18 ms/truy vấn.
