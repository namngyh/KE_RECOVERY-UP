"""S0 - nạp, kiểm toán và gộp dữ liệu VN30F1M.

Đầu vào : ohlc_export.csv (nến 1 phút).
          Cột: SYMBOL, TRADING_DATE (YYYYMMDD), TRADING_TIME (HH:MM:SS, giờ VN, nhãn = đầu phút),
               OPEN_PX, HIGH_PX, LOW_PX, CLOSE_PX, VOL, BUY_VOL, BUY_VAL, SELL_VOL, SELL_VAL.
Đầu ra  : bảng 1m và 5m có cột open_time_ms (UTC ms, đầu nến) như engine QPT mong đợi.

Quy ước (đề cương, mục 4-6):
* Khối lượng không phân loại (VOL > 0 nhưng BUY + SELL = 0, ví dụ ATO/ATC) KHÔNG bị điền 0 ở mức
  mua/bán; cờ `unclassified` được ghi để bước đặc trưng đặt I_t = NaN.
* Không nội suy OHLC, không tạo nến cho phút không có giao dịch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VN_TZ = "Asia/Ho_Chi_Minh"
RAW_COLUMNS = [
    "SYMBOL", "TRADING_DATE", "OPEN_PX", "HIGH_PX", "LOW_PX", "CLOSE_PX", "VOL",
    "TRADING_TIME", "BUY_VOL", "BUY_VAL", "SELL_VOL", "SELL_VAL",
]


def load_1m(path: str) -> pd.DataFrame:
    """Đọc CSV, tạo timestamp VN, open_time_ms (UTC) và tên cột chuẩn. Sắp xếp tăng dần."""
    raw = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    ts = pd.to_datetime(raw["TRADING_DATE"].astype(str) + " " + raw["TRADING_TIME"],
                        format="%Y%m%d %H:%M:%S")
    df = pd.DataFrame({
        "ts": ts,
        "symbol": raw["SYMBOL"],
        "open": raw["OPEN_PX"].astype(float),
        "high": raw["HIGH_PX"].astype(float),
        "low": raw["LOW_PX"].astype(float),
        "close": raw["CLOSE_PX"].astype(float),
        "volume": raw["VOL"].astype("int64"),
        "buy_vol": raw["BUY_VOL"].astype("int64"),
        "sell_vol": raw["SELL_VOL"].astype("int64"),
        "buy_val": raw["BUY_VAL"].astype("int64"),
        "sell_val": raw["SELL_VAL"].astype("int64"),
    })
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["open_time_ms"] = _to_utc_ms(df["ts"])
    return df


def _to_utc_ms(ts: pd.Series) -> pd.Series:
    """Giờ VN (naive) -> epoch ms UTC. VN không có giờ mùa hè nên phép đổi không mơ hồ."""
    utc = ts.dt.tz_localize(VN_TZ).dt.tz_convert("UTC").dt.tz_localize(None)
    return ((utc - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)).astype("int64")


def audit_1m(df: pd.DataFrame) -> dict:
    """Kiểm tra logic dữ liệu 1 phút. Chỉ báo cáo, không sửa dữ liệu."""
    px = df[["open", "high", "low", "close"]]
    classified = df["buy_vol"] + df["sell_vol"]
    minute = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    day = df["ts"].dt.normalize()
    per_day = df.groupby(day).size()
    unclassified = (df["volume"] > 0) & (classified == 0)
    total_vol = max(int(df["volume"].sum()), 1)
    return {
        "rows": int(len(df)),
        "sessions": int(day.nunique()),
        "first_ts": str(df["ts"].iloc[0]),
        "last_ts": str(df["ts"].iloc[-1]),
        "symbols": {k: int(v) for k, v in df["symbol"].value_counts().items()},
        "duplicate_ts": int(df["ts"].duplicated().sum()),
        "null_cells": int(df.isna().sum().sum()),
        "non_positive_price": int((px <= 0).any(axis=1).sum()),
        "high_below_open_or_close": int((df["high"] < df[["open", "close"]].max(axis=1)).sum()),
        "low_above_open_or_close": int((df["low"] > df[["open", "close"]].min(axis=1)).sum()),
        "negative_volume": int((df[["volume", "buy_vol", "sell_vol"]] < 0).any(axis=1).sum()),
        "classified_gt_volume": int((classified > df["volume"]).sum()),
        "classified_eq_volume": int((classified == df["volume"]).sum()),
        "classified_lt_volume_nonzero": int(((classified < df["volume"]) & (classified > 0)).sum()),
        "unclassified_rows": int(unclassified.sum()),
        "unclassified_volume_share": float(df.loc[unclassified, "volume"].sum() / total_vol),
        "classified_volume_share": float(classified.sum() / total_vol),
        "zero_volume_rows": int((df["volume"] == 0).sum()),
        "minute_of_day_min": _hhmm(int(minute.min())),
        "minute_of_day_max": _hhmm(int(minute.max())),
        "rows_before_0900": int((minute < 9 * 60).sum()),
        "rows_from_1445": int((minute >= 14 * 60 + 45).sum()),
        "rows_in_lunch_1130_1259": int(((minute >= 11 * 60 + 30) & (minute < 13 * 60)).sum()),
        "rows_per_session_min": int(per_day.min()),
        "rows_per_session_median": float(per_day.median()),
        "rows_per_session_max": int(per_day.max()),
        "sessions_with_fewer_than_100_rows": int((per_day < 100).sum()),
        "weekend_sessions": int((per_day.index.dayofweek >= 5).sum()),
    }


def _hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def resample(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """Gộp nến 1 phút thành nến `rule`. Nhãn nến = đầu khoảng (floor); không tạo nến rỗng.

    open/close lấy nến 1 phút đầu/cuối có mặt trong khoảng, high/low là max/min,
    khối lượng và giá trị mua/bán là tổng. Cờ `unclassified` = tổng khối lượng > 0 nhưng
    tổng mua + bán = 0.
    """
    g = df.assign(bucket=df["ts"].dt.floor(rule)).groupby("bucket", sort=True)
    out = g.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
        buy_val=("buy_val", "sum"), sell_val=("sell_val", "sum"), n_minutes=("ts", "size"),
    ).reset_index().rename(columns={"bucket": "ts"})
    out["unclassified"] = (out["volume"] > 0) & ((out["buy_vol"] + out["sell_vol"]) == 0)
    out["open_time_ms"] = _to_utc_ms(out["ts"])
    return out[["ts", "open_time_ms", "open", "high", "low", "close", "volume",
                "buy_vol", "sell_vol", "buy_val", "sell_val", "n_minutes", "unclassified"]]


def check_resample(df1: pd.DataFrame, df5: pd.DataFrame) -> dict:
    """Bảo toàn khi gộp: tổng khối lượng, tổng mua/bán, biên giá, số phiên."""
    return {
        "volume_equal": bool(df1["volume"].sum() == df5["volume"].sum()),
        "buy_equal": bool(df1["buy_vol"].sum() == df5["buy_vol"].sum()),
        "sell_equal": bool(df1["sell_vol"].sum() == df5["sell_vol"].sum()),
        "high_max_equal": bool(df1["high"].max() == df5["high"].max()),
        "low_min_equal": bool(df1["low"].min() == df5["low"].min()),
        "minutes_conserved": bool(df5["n_minutes"].sum() == len(df1)),
        "sessions_equal": bool(df1["ts"].dt.normalize().nunique() == df5["ts"].dt.normalize().nunique()),
        "bars_5m": int(len(df5)),
        "bars_with_unclassified": int(df5["unclassified"].sum()),
    }


def session_gaps(df: pd.DataFrame, minutes: int = 5, limit: int = 20) -> pd.DataFrame:
    """Khoảng trống trong phiên (hai nến liên tiếp cùng ngày cách nhau hơn `minutes` phút).

    Chỉ liệt kê để phân loại thủ công (không giao dịch / lỗi thu thập / nghỉ trưa); không điền.
    """
    day = df["ts"].dt.normalize()
    gap = df["ts"].diff().dt.total_seconds().div(60)
    sel = day.eq(day.shift()) & (gap > minutes)
    res = pd.DataFrame({"ts": df["ts"][sel], "gap_minutes": gap[sel]})
    return res.sort_values("gap_minutes", ascending=False).head(limit).reset_index(drop=True)
