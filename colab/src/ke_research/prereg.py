"""S4 - đăng ký trước: mọi tham số, ngưỡng, luật chọn và luật qua cổng (đề cương mục 3.4, 10, 11).

Nguồn sự thật duy nhất là PREREG dưới đây. Bước S4 ghi nó ra results/S4/preregistration.json cùng mã băm SHA-256;
các bước sau gọi verify() để chắc rằng tham số không bị đổi sau khi đã bắt đầu xem kết quả.
Mọi giá trị đề cương chưa ấn định (cửa sổ, epsilon, chân trời, lưới lambda, ngưỡng...) được khai báo ở đây, kèm
ghi chú đó là lựa chọn của nhóm thực hiện chứ không phải của đề cương.
"""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

PREREG = {
    "version": "1.0",
    "date": "2026-11-01",
    "splits": {  # dev: ts < dev_end ; val: dev_end <= ts < val_end ; test: ts >= val_end (niêm phong)
        "dev_end": "2022-01-01", "val_end": "2024-01-01",
        "note": "Dev 11/2017-2021, val 2022-2023, test 01/2024-07/2026. Test chỉ chạm một lần ở S9.",
    },
    "costs": {
        "tax_rate": 0.001, "fee_rate": 0.00005, "slippage_points": 0.2, "initial_margin_rate": 0.1848,
        "contracts": 1, "infra_fee": "không cộng: phí 0,005% được coi là all-in (quyết định của người dùng)",
        "tax_assumption": "tax_rate=0,1% áp cho toàn lịch sử với ký quỹ 18,48% (không dùng tỷ lệ VSDC theo ngày)",
    },
    "utility": {
        "name": "daily_sharpe_net_points", "periods_per_year": 250, "rf_daily": 0.0,
        "definition": "U = sqrt(250) * mean(daily net P&L points) / std(daily net P&L points)",
    },
    "features": {  # lựa chọn của nhóm thực hiện (đề cương mục 6 không ấn định)
        "slot_window": 60, "slot_min_periods": 20, "eps0": 0.1,
        "eps0_note": "eps0=1e-6 ban đầu cho |I~| tới 2e6 ở 2 nến (cửa sổ ô gần như hằng số, đầu mẫu); chọn 0,1 ~ 1/3 độ lệch chuẩn "
                     "điển hình của I theo ô (0,26), dựa trên thang đo của biến, trước mọi kết quả giao dịch",
        "V_window": 60, "V_stat": "mean", "eps_V": 1.0, "ER_n": 20,
        "missing_rule": "Ĩ thiếu (V_sigma=0) -> đặt 0 sau chuẩn hóa, kèm cờ unclassified; ATC KHÔNG bị loại",
        "delta_I_versions": ["within_session", "allow_overnight"],
        "atc_sensitivity": "độ nhạy bổ sung (không có trong đề cương): bỏ ATC khỏi I, A, dI",
    },
    "horizons": {"bars": [1, 3, 6, 12], "primary": 6,
                 "note": "R_exec: vào ở open t+1, ra ở open t+1+h cùng phiên; trừ trượt giá 2x0,2 điểm và phí 2x0,005%"},
    "tests": {"alpha": 0.05, "tercile": 0.333, "cluster": "session", "multiple_testing": "Holm"},
    "lambda_grid_c": [0.1, 0.25, 0.5, 1.0],
    "lambda_note": "lambda = c * std(g0 trên dev); M1 thử cả dấu âm",
    "filters": {
        "M2_band_q": [0.2, 0.8], "M3_upper_q": 0.8, "M7_er_q": [0.3, 0.5, 0.7],
        "M5C_variants": ["no_entry_near_expiry", "no_entry_after_holiday_or_first_day", "both"],
        "rule": "bộ lọc chỉ chặn lệnh MỞ MỚI; vị thế đang mở được theo dõi và đóng bình thường (đề cương 8.2)",
        "quantiles_from": "dev",
    },
    "M8": {"regimes_er_q": [0.33, 0.67], "L_m": {"low": [96, 40], "mid": [64, 29], "high": [32, 20]},
           "controls": [[32, 20], [96, 40]],
           "rule": "chế độ theo ER trung vị của phiên TRƯỚC (cập nhật mỗi phiên); ngưỡng = phân vị của ER trung vị phiên trên dev; "
                   "không ghi nhận giao cắt tại nến đổi chế độ",
           "note": "thiết kế ban đầu (đổi chế độ theo ER từng nến) cho 83% giao cắt giả tại nến đổi chế độ nên được thay trước khi có kết quả giao dịch"},
    "M4": {"model": "logistic L2", "C": 1.0, "retrain": "monthly", "embargo_sessions": 1, "min_events": 100,
           "tau_grid": [0.45, 0.50, 0.55], "label": "net_pnl>0 của lệnh M0 (sau chi phí)",
           "features": ["absB_norm", "dist_ema_z", "I_tilde", "A_tilde", "dI_tilde", "J_tilde", "near_expiry",
                        "after_holiday", "first_day_after_expiry", "session_part", "logh_z", "side"]},
    "egarchx": {
        "min_train_sessions": 250, "refit_every": 60, "distributions": ["normal", "student_t", "skew_t"],
        "dist_selection": "log-score nhỏ nhất trên val",
        "return": "r_cc (đã hiệu chỉnh roll) x100 (phần trăm)", "mean": "hằng số",
        "X": ["log RV_intra(d-1)", "log(1e-3 + r_night(d-1)^2)"], "D": ["near_expiry", "after_holiday", "first_day_after_expiry"],
        "RV": {"chi_lunch": 0, "chi_night": 0, "A_roll": 0, "intra": "tổng bình phương lợi suất 1m trong phiên, bỏ bước qua nghỉ trưa"},
        "benchmarks": ["rolling_var_20", "ewma_0.94", "mean_RV_5"], "var_levels": [0.01, 0.05],
        "gate": "QLIKE nhỏ hơn có ý nghĩa (DM một phía p<0,10) so với đối chứng tốt nhất trên val; nếu không, M1-M4 chỉ là kiểm chứng âm",
    },
    "gate": {
        "paired_bootstrap": {"block_sessions": 5, "reps": 2000, "seed": 20261101},
        "pass_rule": "trên val: dU>0, p một phía Holm<0,10 (họ M1-M8), MDD_val<=1,10 x MDD_val(M0)",
        "holm_alpha": 0.10, "mdd_tolerance": 1.10,
        "selection_within_family": "cấu hình có U_dev cao nhất; giá trị val không dùng để chọn trong họ",
    },
    "overfitting": {"pbo_blocks": 16, "dsr_n_trials": "tổng số cấu hình đã chạy ở S6+S8", "spa_reps": 2000},
    "test_protocol": {"runs": 1, "lock_file": "TEST_LOCK.json",
                      "note": "M0 đã được xem trên test ở bước S1 (benchmark của mã đóng băng); không dùng để chọn gì."},
}


def canonical() -> str:
    return json.dumps(PREREG, sort_keys=True, ensure_ascii=False, indent=1)


def digest() -> str:
    return hashlib.sha256(canonical().encode("utf-8")).hexdigest()


def write(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preregistration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sha256": digest(), "prereg": PREREG}, f, sort_keys=True, ensure_ascii=False, indent=1)
    return path


def verify(root: str) -> None:
    """Báo lỗi nếu PREREG trong mã khác bản đã đăng ký ở S4 (tham số bị đổi sau khi bắt đầu)."""
    path = os.path.join(root, "results", "S4", "preregistration.json")
    if not os.path.exists(path):
        raise AssertionError("Chưa có preregistration.json (chạy S4 trước)")
    saved = json.load(open(path, encoding="utf-8"))["sha256"]
    if saved != digest():
        raise AssertionError("PREREG đã thay đổi sau khi đăng ký; chạy lại từ S4 và ghi nhận việc thay đổi")


def period_of(ts: pd.Series) -> pd.Series:
    """Nhãn 'dev' / 'val' / 'test' theo thời gian."""
    s = PREREG["splits"]
    out = pd.Series("test", index=ts.index)
    out[ts < pd.Timestamp(s["val_end"])] = "val"
    out[ts < pd.Timestamp(s["dev_end"])] = "dev"
    return out
