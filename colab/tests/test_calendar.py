"""Kiểm thử S2 - biến lịch và chuẩn hóa theo ô (đề cương mục 5.1, 5.4). Dữ liệu tổng hợp."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import calendar_feats as cf  # noqa: E402


def _bars_for(days, slots=("09:00", "09:05")):
    ts = [pd.Timestamp(f"{d} {s}") for d in days for s in slots]
    return pd.DataFrame({"ts": ts})


def test_slot_standardize_matches_hand_computation_and_uses_only_past():
    # một ô duy nhất, 6 phiên; cửa sổ 3, tối thiểu 2 quan sát quá khứ
    x = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    slot = pd.Series([540] * 6)
    z = cf.slot_standardize(x, slot, window=3, min_periods=2, eps0=0.0)
    assert z.iloc[:2].isna().all()                       # chưa đủ 2 quan sát trước
    past = [1.0, 2.0]                                    # quan sát thứ 3 (index 2): quá khứ = [1, 2]
    assert z.iloc[2] == pytest.approx((4.0 - np.mean(past)) / np.std(past, ddof=1))
    past = [2.0, 4.0, 8.0]                               # index 4 (16.0): cửa sổ 3 quan sát gần nhất trước nó
    assert z.iloc[4] == pytest.approx((16.0 - np.mean(past)) / np.std(past, ddof=1))


def test_slot_standardize_does_not_leak_future():
    rng = np.random.default_rng(3)
    n_days, slots = 80, [540, 545, 550]
    slot = pd.Series(np.tile(slots, n_days))
    x = pd.Series(rng.normal(size=len(slot)))
    z1 = cf.slot_standardize(x, slot, window=20, min_periods=5)
    x2 = x.copy(); x2.iloc[150:] = x2.iloc[150:] * 5 + 100   # đổi toàn bộ tương lai
    z2 = cf.slot_standardize(x2, slot, window=20, min_periods=5)
    assert np.allclose(z1.iloc[:150].fillna(-1e9), z2.iloc[:150].fillna(-1e9))


def test_slot_standardize_separates_slots():
    x = pd.Series([1.0, 100.0] * 30)                     # ô A luôn ~1, ô B luôn ~100
    slot = pd.Series([540, 545] * 30)
    z = cf.slot_standardize(x, slot, window=10, min_periods=5, eps0=1e-8)
    assert z.dropna().abs().max() < 1e-3                 # mỗi ô chỉ so với chính nó


def test_slot_standardize_ignores_nan_and_requires_min_periods():
    x = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0, 6.0])
    slot = pd.Series([540] * 6)
    z = cf.slot_standardize(x, slot, window=5, min_periods=3, eps0=0.0)
    assert z.isna().all()                                # không bao giờ đủ 3 quan sát hợp lệ trong quá khứ


def test_session_calendar_flags_and_k_d():
    # T6 15/03 -> T2 18/03 (cuối tuần), rồi nghỉ lễ thứ Tư 20/03 (bỏ 1 ngày làm việc)
    days = ["2024-03-14", "2024-03-15", "2024-03-18", "2024-03-19", "2024-03-21", "2024-03-22"]
    bars = _bars_for(days)
    rt = pd.DataFrame({"exp_day": [pd.Timestamp("2024-03-19")], "new_day": [pd.Timestamp("2024-03-21")]})
    cal = cf.session_calendar(bars, rt)
    d = lambda s: pd.Timestamp(s)  # noqa: E731
    assert cal.loc[d("2024-03-18"), "after_weekend"] and not cal.loc[d("2024-03-18"), "after_holiday"]
    assert cal.loc[d("2024-03-21"), "after_holiday"] and cal.loc[d("2024-03-21"), "off_days"] == 1
    assert cal.loc[d("2024-03-21"), "first_day_after_expiry"] and cal["first_day_after_expiry"].sum() == 1
    # đáo hạn 19/03: các phiên còn lại tính theo phiên thực có dữ liệu
    assert [int(cal.loc[d(x), "k_d"]) for x in ("2024-03-14", "2024-03-15", "2024-03-18", "2024-03-19")] == [3, 2, 1, 0]
    assert cal.loc[d("2024-03-14"), "near_expiry"] and cal.loc[d("2024-03-19"), "near_expiry"]
    assert not cal.loc[d("2024-03-14"), "k_estimated"]
    assert cal.loc[d("2024-03-22"), "k_estimated"]       # sau roll cuối: ước theo quy tắc, có cờ


def test_add_bar_calendar_positions():
    bars = _bars_for(["2024-03-18", "2024-03-19"], slots=("09:00", "09:05", "09:10"))
    cal = cf.session_calendar(bars, pd.DataFrame({"exp_day": [], "new_day": []}))
    out = cf.add_bar_calendar(bars, cal)
    assert list(out["slot"][:3]) == [540, 545, 550] and list(out["bar_of_day"]) == [0, 1, 2, 0, 1, 2]


def test_classify_bar_gaps_counts_missing_slots():
    days = [f"2024-03-{d:02d}" for d in (18, 19, 20, 21)]
    bars = _bars_for(days, slots=("09:00", "09:05", "09:10"))
    bars = bars.drop(index=[4])                          # phiên 20/03 thiếu ô 09:05
    res = cf.classify_bar_gaps(bars)
    assert res["summary"]["expected_slots_per_session"] == 3
    assert res["summary"]["missing_slots_total"] == 1 and res["summary"]["sessions_with_any_missing"] == 1
    assert "Không phân biệt" in res["summary"]["note"]
