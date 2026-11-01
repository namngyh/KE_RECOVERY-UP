# Báo cáo thí nghiệm KE Recovery (VN30F1M)

*Tạo tự động; mã băm đăng ký trước: `fca2b265e4921e08`. Mọi kết quả là điểm P&L ròng sau phí, trượt giá và thuế 0,1%, 1 hợp đồng.*

## 1. Tóm tắt

- **M0 (KE gốc)**: U (Sharpe ngày) dev 3.54, val 1.49; lợi nhuận ròng dev 2573, val 605 điểm.
- **Cổng EGARCH-X**: KHÔNG QUA (QLIKE val 1.2883 so với đối chứng tốt nhất mean_RV_5 1.2080, DM p = 1.000).
- **Cải tiến qua cổng trên val**: KHÔNG CÓ.
- **M9**: M9 không dựng: chỉ 0 cải tiến qua cổng (cần >= 2 để ghép).
- **PBO** (dev+val, 73 cấu hình) = 0.28; **SPA val** p = 0.371.

- **Tập test (chạy một lần)**: M0: U 1.18, 641 điểm.

## 2. Dữ liệu và kiểm toán (S0-S2)

- 525,635 nến 1 phút, 2171 phiên (2017-11-06 - 2026-07-17); nến chưa phân loại mua/bán: 1,457 (1.49% khối lượng).
- 5m gộp từ 1m khớp giá file 5m gốc: True; lệch khối lượng ở 15 nến.
- Tái lập báo cáo gốc (thuế 0): 1867 lệnh, 4337.4 điểm, profit factor 1.66.
- 105 lần roll; Delta ước lượng bằng chỉ số VN30 (xấp xỉ): trung bình -1.52, độ lệch chuẩn 11.83 điểm; lần roll bất thường: 2020-05-21.
- Độ lệch chuẩn lợi suất qua đêm ngày roll: 0.0130 (giá thực) -> 0.0069 (đã hiệu chỉnh).

## 3. Đặc trưng (S3)

- ATC: độ lệch chuẩn I = 0.88 so với 0.30 ở nến thường; sau chuẩn hóa theo ô 1.08 so với 0.74.

## 4. Kiểm định giả thuyết (S5, chân trời 6 nến)

| test | period | p | p_holm |
|---|---|---|---|
| H1 | dev | 0.9590 | 0.959 |
| H1 | val | 0.7721 | 1.000 |
| H2 | dev | 0.0321 | 0.128 |
| H2 | val | 0.2962 | 1.000 |
| H3_in | dev | 0.1697 | 0.339 |
| H3_in | val | 0.3736 | 1.000 |
| H3_ov | dev | 0.0755 | 0.227 |
| H3_ov | val | 0.2893 | 1.000 |
| H4 | dev | 0.0020 | 0.010 |
| H4 | val | 0.2596 | 1.000 |

IC (Spearman, trung bình theo phiên): I dev: 0.043 (t=9.2); dI_in dev: 0.003 (t=1.3); J dev: 0.036 (t=5.3); I val: 0.025 (t=3.6); dI_in val: 0.012 (t=3.1); J val: 0.014 (t=1.4)

H5 (phụ thuộc phương sai dự báo): dev: p=0.282; val: p=0.957

## 5. Ablation M0 (S6)

| config | dev_U | val_U | dev_net_points | val_net_points | dev_trades |
|---|---|---|---|---|---|
| A0_KE_goc | 3.54 | 1.49 | 2573 | 605 | 847 |
| A1_RSI_only | 3.27 | 1.75 | 2361 | 704 | 939 |
| A2_Stoch_only | 3.04 | 1.33 | 2203 | 531 | 841 |
| A3_bo_EMA | 3.21 | 1.21 | 2935 | 621 | 1263 |
| A6_long_only | 2.50 | 0.90 | 1180 | 216 | 492 |
| A6_short_only | 2.39 | 1.17 | 1394 | 389 | 355 |
| A7_dong_cuoi_phien | 2.57 | 1.17 | 1632 | 437 | 1369 |
| A4_wma_L32 | 2.41 | 1.60 | 1810 | 631 | 976 |
| A4_wma_L48 | 3.12 | 1.45 | 2273 | 607 | 850 |
| A4_wma_L64 | 2.81 | 0.86 | 2089 | 362 | 762 |
| A4_wma_L96 | 2.35 | 0.84 | 1753 | 363 | 702 |
| A4_sma_L32 | 3.20 | 1.31 | 2276 | 538 | 849 |
| A4_sma_L48 | 2.67 | 0.51 | 1933 | 219 | 713 |
| A4_sma_L64 | 1.03 | -0.04 | 788 | -17 | 680 |
| A4_sma_L96 | 1.72 | 0.67 | 1391 | 285 | 615 |
| A5_ema_length_266 | 3.49 | 1.51 | 2537 | 625 | 865 |
| A5_ema_length_326 | 3.50 | 1.56 | 2544 | 633 | 834 |
| A5_period_24 | 3.42 | 1.62 | 2465 | 653 | 872 |
| A5_period_30 | 3.48 | 1.42 | 2548 | 574 | 833 |
| A5_k_smooth_18 | 3.55 | 1.44 | 2566 | 586 | 852 |
| A5_k_smooth_22 | 3.47 | 1.62 | 2533 | 660 | 843 |
| A5_reg_period_58 | 3.50 | 2.22 | 2453 | 862 | 882 |
| A5_reg_period_70 | 3.22 | 1.29 | 2348 | 528 | 812 |
| A5_n1_5.94 | 3.48 | 1.41 | 2543 | 575 | 849 |
| A5_n1_7.26 | 3.55 | 1.61 | 2584 | 650 | 851 |
| A5_ma_period_26 | 3.52 | 1.56 | 2545 | 627 | 868 |
| A5_ma_period_32 | 3.50 | 1.34 | 2551 | 545 | 837 |

## 6. EGARCH-X (S7)

Phân phối chọn theo log-score val: **skew_t**.

| dist | val_qlike | val_log_score | val_pit_ks_p | val_var01_rate | val_var05_rate |
|---|---|---|---|---|---|
| normal | 1.2844 | 1.5939 | 0.0001 | 0.0120 | 0.0361 |
| student_t | 1.2853 | 1.5417 | 0.0363 | 0.0060 | 0.0482 |
| skew_t | 1.2883 | 1.5413 | 0.0112 | 0.0040 | 0.0361 |

Đối chứng (QLIKE val): rolling_var_20 1.2901; ewma_0.94 1.2724; mean_RV_5 1.2080

## 7. M1-M9: cổng trên val (S8-S9)

| family | config | eligible | dU_val | p | p_holm | mdd_ratio | passed |
|---|---|---|---|---|---|---|---|
| M1 | M1_c+0.5 | False | 0.106 | 0.163 | 1.000 | 0.926 | False |
| M2 | M2_band | False | 0.095 | 0.442 | 1.000 | 0.535 | False |
| M3 | M3_upper | False | 0.509 | 0.119 | 0.956 | 0.533 | False |
| M4 | M4_tau0.45 | False | -0.216 | 0.883 | 1.000 | 1.067 | False |
| M5A | M5A_adj | True | 0.010 | 0.460 | 1.000 | 1.075 | False |
| M5B | M5B_season | True | 0.195 | 0.091 | 0.823 | 0.998 | False |
| M5C | M5C_no_entry_after_holiday_or_first_day | True | -0.281 | 0.899 | 1.000 | 0.974 | False |
| M6 | M6_G3ov_c0.1 | True | 0.068 | 0.048 | 0.485 | 1.013 | False |
| M7 | M7_q0.5 | True | -0.023 | 0.525 | 1.000 | 0.859 | False |
| M8 | M8_adaptive | True | -0.219 | 0.766 | 1.000 | 1.730 | False |


Chẩn đoán S8: `{"M4_meta": {"months_with_model": 97, "events_total": 1807, "first_month_with_model": "2018-07", "dev": {"n": 734, "brier": 0.2573107779672299, "brier_base": 0.24910163413493308, "auc": 0.530635967363362, "mean_net_points_by_prob_tercile": [1.2925445067754195, 3.441483680655648, 2.1981264568162358], "win_rate_by_prob_tercile": [0.4163265306122449, 0.5122950819672131, 0.4816326530612245]}, "val": {"n": 429, "brier": 0.2446554655413047, "brier_base": 0.24724925424226132, "auc": 0.565225035161744, "mean_net_points_by_prob_tercile": [1.747728551608292, 2.197793653706213, 0.2851242528670542], "win_rate_by_prob_tercile": [0.3706293706293706, 0.48951048951048953, 0.4825174825174825]}}, "M8_regime_artifacts": {"M8_crosses_raw": 3433, "M8_crosses_at_regime_change": 426}, "M6_best_excluding_ATC_sensitivity": {"config": "M6_G3ov_c0.1", "dev_U": 3.5188567114307254, "val_U": 1.527380993508732, "note": "độ nhạy bổ sung (không có trong đề cương)"}, "position_change_share_first2_bars": {"M0": 0.05684995340167754, "M1": 0.056025039123630674, "M2": 0.08935974205435283, "M3": 0.0669090909090909}}`

## 8. Chống quá khớp

- PBO = 0.28 trên 73 cấu hình (12870 tổ hợp CSCV).
- SPA (val, 10 mô hình so với M0): t = 1.63, p = 0.371.
- DSR (val): M1: 0.88; M2: 0.89; M3: 0.97; M4: 0.75; M5A: 0.84; M5B: 0.90; M5C: 0.72; M6: 0.86; M7: 0.83; M8: 0.74; M0: 0.84

## 9. Tập test (chạy một lần, có khóa)

| model | U | net_points | mdd | trades | dU_vs_M0 | p | DSR |
|---|---|---|---|---|---|---|---|
| M0 | 1.178 | 640.9 | 314.1 | 531 |  |  | 0.708 |


## 10. Giới hạn và điều cần thừa nhận

- Delta của lần roll là ước lượng bằng chỉ số VN30 (basis), không đo tại thời điểm chồng lấn như đề cương mô tả; có một lần roll bất thường (21/05/2020).
- Thuế 0,1% áp cho toàn lịch sử với ký quỹ 18,48%; Thông tư 87/2026 chỉ hiệu lực từ 01/07/2026 và tỷ lệ VSDC thật thay đổi theo ngày. Phí 0,005% được coi là all-in, không cộng phí hạ tầng.
- M0 đã được xem trên tập test ở bước S1 (benchmark của mã đóng băng); không dùng để chọn gì.
- Các tham số đề cương chưa ấn định (cửa sổ, epsilon, lưới lambda, ngưỡng, luật chế độ M8...) là lựa chọn của nhóm thực hiện, ghi trong preregistration.json; hai tham số (eps0 và luật M8) được chọn lại sau khi thấy lỗi thang đo/giao cắt giả, trước mọi kết quả giao dịch.
- M4 học trên lệnh M0 thật nhưng áp cho mọi nến ứng viên (xấp xỉ). Nến ATC không bị loại ở cấu hình chính (độ nhạy loại ATC là bổ sung của nhóm thực hiện).
- Không phân biệt được nến thiếu do 'không khớp lệnh' hay 'lỗi thu thập' khi chỉ có OHLCV.
