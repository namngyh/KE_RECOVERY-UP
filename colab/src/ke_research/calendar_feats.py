"""S2 - biến lịch giao dịch và chuẩn hóa theo ô thời gian (đề cương mục 5.1, 5.4).

5.1  Phân loại nến thiếu: dữ liệu chỉ có OHLCV nên KHÔNG phân biệt được 'không có khớp lệnh thực' với 'lỗi thu thập';
     chỉ đếm số ô thời gian dự kiến bị thiếu theo phiên và đánh dấu phiên thiếu nhiều. Không nội suy, không điền 0.
5.4  k_d = số phiên còn lại đến ngày đáo hạn của hợp đồng đang giao dịch; 1(k_d in {0,1,2,3}); ngày đầu sau đáo hạn;
     ngày sau kỳ nghỉ; số ngày nghỉ; vị trí nến trong phiên; chuẩn hóa theo lịch (công thức seasonal):
         X~ = (X - mu_{tau, d-1}) / (sigma_{tau, d-1} + eps0)   với mu, sigma chỉ từ các phiên TRƯỚC d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.analysis.vn30f import expiry as _third_thursday_expiry
from backend.analysis.vn30f import front_month


def slot_minute(ts: pd.Series) -> pd.Series:
    """Ô thời gian của nến 5m = phút trong ngày của đầu nến (ví dụ 09:00 -> 540)."""
    return (ts.dt.hour * 60 + ts.dt.minute).astype("int64")


def expected_slots(bars: pd.DataFrame, min_share: float = 0.5) -> list[int]:
    """Các ô thời gian xuất hiện trong ít nhất `min_share` số phiên (xác định từ dữ liệu, không giả định lịch)."""
    day = bars["ts"].dt.normalize()
    n_days = day.nunique()
    counts = slot_minute(bars["ts"]).groupby(day.to_numpy()).apply(lambda s: set(s)).explode().value_counts()
    return sorted(int(s) for s, c in counts.items() if c / n_days >= min_share)


def classify_bar_gaps(bars: pd.DataFrame, min_share: float = 0.5, heavy_missing_share: float = 0.25) -> dict:
    """Đếm ô dự kiến bị thiếu mỗi phiên. Trả về tóm tắt và bảng theo phiên (cột missing, share)."""
    slots = expected_slots(bars, min_share)
    day = bars["ts"].dt.normalize()
    sl = slot_minute(bars["ts"])
    present = pd.Series(sl.isin(slots).to_numpy(), index=bars.index).groupby(day).sum()
    per = pd.DataFrame({"present_expected": present})
    per["missing"] = len(slots) - per["present_expected"]
    per["share"] = per["missing"] / len(slots)
    per["heavy"] = per["share"] >= heavy_missing_share
    summary = {
        "expected_slots_per_session": len(slots),
        "expected_slot_minutes": [f"{s // 60:02d}:{s % 60:02d}" for s in slots],
        "missing_slots_total": int(per["missing"].sum()),
        "missing_share_overall": float(per["missing"].sum() / (len(slots) * len(per))),
        "sessions_with_any_missing": int((per["missing"] > 0).sum()),
        "sessions_heavy_missing": int(per["heavy"].sum()),
        "note": "Không phân biệt được 'không khớp lệnh' và 'lỗi thu thập' nếu chỉ có OHLCV.",
    }
    return {"summary": summary, "per_session": per}


def session_calendar(bars: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    """Biến lịch theo phiên. `rt` là bảng roll (cột exp_day, new_day)."""
    days = pd.DatetimeIndex(sorted(bars["ts"].dt.normalize().unique()))
    pos = pd.Series(np.arange(len(days)), index=days)
    cal = pd.DataFrame(index=days)
    cal.index.name = "day"
    prev = days.to_series().shift(1)
    gap = (days.to_series() - prev).dt.days
    cal["off_days"] = (gap - 1).astype("float")
    skipped = [np.nan] + [int(np.busday_count((prev.iloc[i] + pd.Timedelta(days=1)).date(), days[i].date()))
                          for i in range(1, len(days))]
    cal["skipped_weekdays"] = skipped
    cal["after_holiday"] = cal["skipped_weekdays"] > 0
    cal["after_weekend"] = (days.dayofweek == 0) & (cal["off_days"] == 2)

    exp_days = pd.DatetimeIndex(rt["exp_day"]) if len(rt) else pd.DatetimeIndex([])
    new_days = set(pd.DatetimeIndex(rt["new_day"])) if len(rt) else set()
    k, est = [], []
    for d in days:
        nxt = exp_days[exp_days >= d]
        if len(nxt):
            k.append(int(pos[nxt[0]] - pos[d])); est.append(False)
        else:  # sau lần roll cuối trong dữ liệu: ngày đáo hạn theo quy tắc thứ Năm thứ ba, đếm ngày làm việc
            y, m = front_month(d.date())
            k.append(int(np.busday_count(d.date(), _third_thursday_expiry(y, m)))); est.append(True)
    cal["k_d"] = k
    cal["k_estimated"] = est
    cal["near_expiry"] = cal["k_d"].isin([0, 1, 2, 3])
    cal["first_day_after_expiry"] = [d in new_days for d in days]
    return cal


def add_bar_calendar(bars: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Gắn biến lịch phiên vào từng nến và thêm vị trí nến: slot (phút trong ngày) và bar_of_day (thứ tự trong phiên)."""
    day = bars["ts"].dt.normalize()
    out = pd.DataFrame(index=bars.index)
    out["day"] = day
    out["slot"] = slot_minute(bars["ts"])
    out["bar_of_day"] = bars.groupby(day).cumcount()
    return out.join(cal, on="day")


def slot_baseline(x: pd.Series, slot: pd.Series, window: int = 60, min_periods: int = 20,
                  stat: str = "mean") -> pd.Series:
    """Trung bình (hoặc trung vị) của `window` quan sát TRƯỚC ĐÓ trong cùng ô thời gian (không gồm quan sát hiện tại).

    Dùng cho V-bar trong A_t (đề cương mục 6.2) và cho mu của công thức seasonal."""
    if stat not in ("mean", "median"):
        raise ValueError("stat phải là 'mean' hoặc 'median'")
    past = x.groupby(slot).shift(1)
    roll = past.groupby(slot).rolling(window, min_periods=min_periods)
    return getattr(roll, stat)().reset_index(level=0, drop=True).sort_index()


def slot_standardize(x: pd.Series, slot: pd.Series, window: int = 60, min_periods: int = 20,
                     eps0: float = 1e-8, stat: str = "mean") -> pd.Series:
    """X~ = (X - mu) / (sigma + eps0), với mu và sigma của CÙNG ô thời gian trong `window` quan sát TRƯỚC đó
    (shift(1) trong từng ô nên không bao giờ dùng giá trị hiện tại hay tương lai).

    stat = "mean" hoặc "median" cho mu (đề cương cho phép trung bình hoặc trung vị). Quan sát NaN bị bỏ qua;
    cần ít nhất `min_periods` quan sát hợp lệ, nếu không kết quả là NaN. Mỗi ô có tối đa một quan sát mỗi phiên.
    """
    if stat not in ("mean", "median"):
        raise ValueError("stat phải là 'mean' hoặc 'median'")
    past = x.groupby(slot).shift(1)
    roll = past.groupby(slot).rolling(window, min_periods=min_periods)
    mu = getattr(roll, stat)().reset_index(level=0, drop=True).sort_index()
    sd = roll.std(ddof=1).reset_index(level=0, drop=True).sort_index()
    return (x - mu) / (sd + eps0)
