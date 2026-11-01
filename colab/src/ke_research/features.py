"""S3 - đặc trưng khối lượng mua/bán chủ động (đề cương mục 6).

I_t  = (V+ - V-) / (V+ + V-)      thuộc [-1, 1]; V+ + V- = 0 -> NaN + cờ unclassified   (công thức imb)
A_t  = log(1 + V_t / (Vbar_{d-1,tau} + eps_V))  Vbar = trung bình/trung vị cùng ô thời gian ở W_V quan sát trước
dI_t = I_t - I_{t-1}   hai phiên bản: 'in' (chỉ trong cùng phiên) và 'ov' (cho phép qua đêm)
J_t  = q_t^KE * I_t    (đồng thuận với hướng KE; dựng ở bước dùng vì cần vị thế q)
ER_t = |C_t - C_{t-n}| / sum_{j<n} |C_{t-j} - C_{t-j-1}|   (efficiency ratio, M7/M8)
Các biến được chuẩn hóa theo công thức seasonal (calendar_feats.slot_standardize): chỉ dùng phiên trước.
Quy tắc thiếu (khai báo trong prereg): sau chuẩn hóa, NaN -> 0 trong *_f, cờ unclassified được giữ riêng.
Nến ATC không bị loại ở cấu hình chính; exclude_atc=True là độ nhạy bổ sung (không có trong đề cương).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import calendar_feats as cf
from .prereg import PREREG

ATC_SLOT = 14 * 60 + 45


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n, min_periods=n).sum()
    return (num / den.replace(0.0, np.nan)).clip(0.0, 1.0)


def build_features(bars: pd.DataFrame, exclude_atc: bool = False, params: dict | None = None) -> pd.DataFrame:
    """`bars` cần: ts, volume, buy_vol, sell_vol, close (giá dùng cho ER: chuỗi hiệu chỉnh)."""
    p = params or PREREG["features"]
    slot = cf.slot_minute(bars["ts"])
    day = bars["ts"].dt.normalize()
    sigma = (bars["buy_vol"] + bars["sell_vol"]).astype(float)
    I = ((bars["buy_vol"] - bars["sell_vol"]) / sigma.where(sigma > 0)).astype(float)
    V = bars["volume"].astype(float)
    unclassified = (V > 0) & (sigma == 0)
    if exclude_atc:
        atc = slot == ATC_SLOT
        I = I.mask(atc)
        V = V.mask(atc)
    vbar = cf.slot_baseline(V, slot, p["V_window"], p["slot_min_periods"], p["V_stat"])
    A = np.log1p(V / (vbar + p["eps_V"]))
    dI_ov = I - I.shift(1)
    dI_in = dI_ov.where(day == day.shift(1))
    out = pd.DataFrame({"slot": slot, "day": day, "unclassified": unclassified.to_numpy(),
                        "I": I, "A": A, "dI_in": dI_in, "dI_ov": dI_ov})
    for c in ("I", "A", "dI_in", "dI_ov"):
        z = cf.slot_standardize(out[c], slot, p["slot_window"], p["slot_min_periods"], p["eps0"])
        out[f"{c}_t"] = z                      # Ĩ: có thể NaN
        out[f"{c}_tf"] = z.fillna(0.0)         # NaN -> 0 (quy tắc thiếu đã đăng ký)
    out["ER"] = efficiency_ratio(bars["close"], p["ER_n"])
    return out
