# Tinh chỉnh khối hợp nhất MagicLens-base trên FashionIQ dress train

Train: 5441 truy vấn, 10878 ảnh (backbone CLIP đóng băng; học 16.8M tham số). Đánh giá: dev 930 / test 931 truy vấn của tập val, như fiq_experiments.py. Chọn tốc độ học và epoch theo R_avg của dev.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| MagicLens zero-shot (baseline) | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| Tinh chỉnh (lr=1e-05, epoch 14) | 45.3 | 11.5 | 25.7 | 34.2 | 57.9 | 46.0 |

Chênh lệch trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): R@10 +10.8 [+8.3, +13.3], R_avg +13.3 [+11.2, +15.5] điểm.

## Đường học (dev R_avg theo epoch)

| Epoch | lr=1e-05 | lr=3e-05 | lr=0.0001 |
|---|---|---|---|
| 1 | 39.5 | 39.1 | 31.7 |
| 2 | 41.2 | 40.4 | 36.8 |
| 3 | 41.6 | 41.8 | 37.0 |
| 4 | 43.7 | 43.7 | 38.9 |
| 5 | 43.8 | 43.1 | 38.2 |
| 6 | 43.7 | 43.5 | 39.9 |
| 7 | 44.1 | 44.0 | 40.2 |
| 8 | 44.1 | 44.5 | 39.8 |
| 9 | 43.4 | 43.4 | 39.0 |
| 10 | 44.6 | 42.8 | 38.8 |
| 11 | 44.9 | 43.3 | 38.3 |
| 12 | 44.0 | 43.8 | 37.9 |
| 13 | 44.6 | 42.5 | 37.6 |
| 14 | 45.3 | 41.3 | 36.3 |
| 15 | 44.6 | 39.8 | 37.0 |

Thời gian huấn luyện (CPU, mọi lần thử, gồm đánh giá mỗi epoch): 35.2 phút.
