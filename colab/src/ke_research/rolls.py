"""S2 - chuyển hợp đồng và chuỗi giá hiệu chỉnh (đề cương mục 5.2, 5.3).

Đề cương:  P_adj = P_raw + A_t,  A_t = sum_{d* > d(t)} Delta_{d*}   (cộng lùi, neo vào hợp đồng mới nhất)
           Delta_{d*} = P_{b,d*} - P_{a,d*}   (chênh lệch hai hợp đồng tại thời điểm chồng lấn)
           "Với backtest nhân quả, không được dùng chênh lệch từ lần roll chưa xảy ra".

Triển khai:
* Chuỗi nhân quả dùng điều chỉnh THUẬN:  P_fwd_t = P_raw_t - sum_{d* <= d(t)} Delta_{d*}.
  P_fwd và P_adj của đề cương chỉ khác nhau một hằng số (tổng Delta), nên mọi chỉ báo bất biến theo
  phép tịnh tiến (RSI trên hiệu giá, %K trong cửa sổ, so sánh close với EMA) cho đúng cùng giá trị, trong khi
  P_fwd không bao giờ dùng Delta của lần roll chưa xảy ra. Tính chất này được kiểm tra bằng test.
* Delta KHÔNG đo được trực tiếp vì dữ liệu chỉ có chuỗi liên tục VN30F1M. Ước lượng bằng chỉ số VN30
  (phương pháp cơ sở, như roll_gaps của QPT):
      Delta = [F_close(d*+1) - I_close(d*+1)] - [F_close(d*) - I_close(d*)]
  với F là đóng cửa phiên của chuỗi liên tục, I là đóng cửa chỉ số; biến động thị trường bị triệt tiêu.
  Giả định: cơ sở của hợp đồng mới trôi không đáng kể trong một phiên. Đây là xấp xỉ, không phải Delta
  đo tại thời điểm chồng lấn như đề cương mô tả.
* Giá thực P_raw luôn dùng cho P&L và khớp lệnh; P_fwd chỉ dùng để tạo tín hiệu.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.analysis.vn30f import roll_indices

PRICE_COLS = ["open", "high", "low", "close"]


def load_index_daily(path: str) -> pd.Series:
    """Đóng cửa chỉ số VN30 theo ngày (index = ngày chuẩn hóa)."""
    d = pd.read_csv(path)
    day = pd.to_datetime(d["datetime_vn"]).dt.normalize()
    s = pd.Series(d["close"].to_numpy(dtype=float), index=day, name="index_close")
    if s.index.duplicated().any():
        raise ValueError("Chuỗi chỉ số có ngày trùng")
    return s.sort_index()


def session_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Mỗi phiên: giá mở (nến đầu), giá đóng (nến cuối), chỉ số nến đầu/cuối, số nến."""
    day = bars["ts"].dt.normalize()
    g = bars.groupby(day)
    out = pd.DataFrame({
        "open": g["open"].first(), "close": g["close"].last(),
        "first_ts": g["ts"].min(), "last_ts": g["ts"].max(), "n_bars": g.size(),
        "first_idx": g.apply(lambda x: x.index[0], include_groups=False),
        "last_idx": g.apply(lambda x: x.index[-1], include_groups=False),
    })
    out.index.name = "day"
    return out


def roll_table(bars: pd.DataFrame, index_close: pd.Series, label: str = "F1M",
               outlier_z: float = 5.0) -> pd.DataFrame:
    """Bảng các lần roll và Delta ước lượng bằng chỉ số.

    Cột: roll_idx (nến đầu tiên của hợp đồng mới), exp_day, new_day, jump (open mới - close cũ),
    basis_exp, basis_next, delta (công thức chính, KHÔNG chỉnh sửa), outlier (|basis_exp| bất thường
    theo z mạnh dựa trên MAD), delta_robust (basis_exp := 0 ở lần roll bị gắn cờ, do hợp đồng hội tụ về chỉ số
    khi đáo hạn; chỉ dùng cho độ nhạy).
    """
    idx = roll_indices(bars["open_time_ms"].to_numpy(), label)
    day = bars["ts"].dt.normalize()
    sess = session_table(bars)
    rows = []
    for i in idx:
        d0, d1 = day.iloc[i - 1], day.iloc[i]
        if d0 not in index_close.index or d1 not in index_close.index:
            raise ValueError(f"Thiếu chỉ số VN30 cho roll {d0.date()} -> {d1.date()}")
        b0 = sess.loc[d0, "close"] - index_close[d0]
        b1 = sess.loc[d1, "close"] - index_close[d1]
        rows.append({"roll_idx": int(i), "exp_day": d0, "new_day": d1,
                     "jump": float(bars["open"].iloc[i] - bars["close"].iloc[i - 1]),
                     "basis_exp": float(b0), "basis_next": float(b1), "delta": float(b1 - b0)})
    rt = pd.DataFrame(rows)
    if rt.empty:
        return rt.assign(outlier=pd.Series(dtype=bool), delta_robust=pd.Series(dtype=float))
    med = rt["basis_exp"].median()
    mad = (rt["basis_exp"] - med).abs().median() * 1.4826
    rt["outlier"] = ((rt["basis_exp"] - med).abs() / max(mad, 1e-9)) > outlier_z
    rt["delta_robust"] = rt["basis_next"] - np.where(rt["outlier"], 0.0, rt["basis_exp"])
    return rt


def cumulative_offset(n: int, roll_idx, delta) -> np.ndarray:
    """sum_{d* <= t} Delta: bằng 0 trước roll đầu tiên, tăng thêm Delta tại nến đầu của hợp đồng mới."""
    cum = np.zeros(n)
    for i, dl in zip(roll_idx, delta):
        cum[int(i):] += float(dl)
    return cum


def forward_adjust(bars: pd.DataFrame, rt: pd.DataFrame, delta_col: str = "delta") -> pd.DataFrame:
    """Chuỗi hiệu chỉnh thuận, nhân quả: P_fwd = P_raw - sum_{d* <= t} Delta. Giữ cột raw_* và cum_offset."""
    out = bars.copy()
    cum = cumulative_offset(len(bars), rt["roll_idx"], rt[delta_col]) if len(rt) else np.zeros(len(bars))
    for c in PRICE_COLS:
        out[f"raw_{c}"] = bars[c].to_numpy()
        out[c] = bars[c].to_numpy() - cum
    out["cum_offset"] = cum
    return out


def asof_back_adjusted(bars: pd.DataFrame, rt: pd.DataFrame, t: int, delta_col: str = "delta") -> pd.DataFrame:
    """Chuỗi như đề cương P_adj nhưng nhân quả: tại thời điểm t chỉ dùng các roll đã xảy ra (roll_idx <= t),
    neo vào hợp đồng đang giao dịch tại t. Dùng để kiểm chứng tương đương với forward_adjust."""
    prefix = bars.iloc[: t + 1].copy()
    cum_back = np.zeros(t + 1)
    for i, dl in zip(rt["roll_idx"], rt[delta_col]):
        if i <= t:
            cum_back[: int(i)] += float(dl)  # A_s = sum_{s < d* <= t} Delta
    for c in PRICE_COLS:
        prefix[c] = prefix[c].to_numpy() + cum_back
    return prefix


def intraday_overnight_returns(bars_adj: pd.DataFrame) -> pd.DataFrame:
    """Mục 5.3: r_intra = log(C_d/O_d), r_night = log(O_d/C_{d-1}), r_cc = r_night + r_intra, theo phiên.

    Tính trên chuỗi đã điều chỉnh roll (đề cương: 'sau khi giải quyết vấn đề hợp đồng'); thêm r_night_raw trên
    giá thực để thấy phần bước nhảy do roll. Phiên đầu tiên không có r_night."""
    adj = session_table(bars_adj)
    raw = session_table(bars_adj.assign(open=bars_adj["raw_open"], close=bars_adj["raw_close"]))
    out = pd.DataFrame(index=adj.index)
    out["r_intra"] = np.log(adj["close"] / adj["open"])
    out["r_night"] = np.log(adj["open"] / adj["close"].shift(1))
    out["r_cc"] = out["r_night"] + out["r_intra"]
    out["r_night_raw"] = np.log(raw["open"] / raw["close"].shift(1))
    return out
