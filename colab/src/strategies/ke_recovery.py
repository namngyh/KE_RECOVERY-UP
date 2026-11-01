"""KE Recovery  (chuyển từ KE_RECOVERY.afl).

Lõi KE: f = LinReg(n1*RSI + StochK), s = WMA(f);  giao cắt f/s + bộ lọc EMA (nến liền trước).
  Mua      : trạng thái "sau giao cắt lên" VÀ close > EMA[t-1]  (vào ở nến đầu tiên thỏa)
  Bán khống: trạng thái "sau giao cắt xuống" VÀ close < EMA[t-1]
  Thoát    : giao cắt ngược chiều (đảo chiều tại cùng nến nếu thỏa điều kiện vào).
Khác biệt so với AFL: AFL khớp tại close của nến tín hiệu; ở đây engine khớp ở open nến kế tiếp.
Không có phí/slippage trong file này (engine xử lý).
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Các phép biến đổi cơ sở (tương đương hàm AmiBroker). Mọi phép đều nhân quả:
# giá trị tại nến i chỉ dùng dữ liệu tới nến i.
# ---------------------------------------------------------------------------
def _rsi(close, n):
    """RSI Wilder: 100*U/(U+D), U và D là EMA (hệ số 1/n) của phần tăng và phần giảm."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 * up / (up + dn).replace(0.0, np.nan)


def _stoch_k(high, low, close, n, smooth):
    """%K Stochastic: vị trí của giá trong biên [thấp nhất, cao nhất] n nến, làm mượt SMA(smooth)."""
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    raw = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    return raw.rolling(smooth).mean()


def _wsum(x, w_lag):
    """out[t] = sum_j w_lag[j] * x[t-j]  (j = số nến lùi về trước, j=0 là nến hiện tại)."""
    a = np.asarray(x, dtype=float)
    L = len(w_lag)
    out = np.full(len(a), np.nan)
    if len(a) >= L:
        win = np.lib.stride_tricks.sliding_window_view(a, L)
        out[L - 1:] = win @ w_lag[::-1]
    return pd.Series(out, index=x.index)


def _linreg(x, r):
    """Giá trị tại ĐIỂM CUỐI của đường hồi quy bình phương tối thiểu trên r quan sát gần nhất."""
    j = np.arange(r)
    w = 2.0 * (2 * r - 1 - 3 * j) / (r * (r + 1))
    return _wsum(x, w)


def _wma(x, m):
    """Trung bình động có trọng số tuyến tính (nến mới nhất có trọng số m)."""
    w = (m - np.arange(m)).astype(float)
    return _wsum(x, w / w.sum())


def _ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def _cross(a, b):
    """Cross(a, b) của AmiBroker: a vượt lên trên b tại nến này."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros(len(a), dtype=bool)
    with np.errstate(invalid="ignore"):
        out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def _flip(on, off):
    """Flip(on, off) của AmiBroker: trạng thái bật từ nến 'on' đến trước nến 'off'."""
    out = np.zeros(len(on), dtype=bool)
    s = False
    for i in range(len(on)):
        if on[i]:
            s = True
        elif off[i]:
            s = False
        out[i] = s
    return out


def _kf(df, period, k_smooth, reg_period, n1):
    """f = n1*LinReg(RSI) + LinReg(StochK) = LinReg(n1*RSI + StochK) (hồi quy là phép tuyến tính)."""
    g = n1 * _rsi(df["close"], period) + _stoch_k(df["high"], df["low"], df["close"], period, k_smooth)
    return _linreg(g, reg_period)


STRATEGY = {
    "name": "KE Recovery",
    "side": "both",               # long | short | both
    "params": {
        "ema_length": {"type": "int", "default": 296, "min": 100, "max": 300},
        "period": {"type": "int", "default": 27, "min": 10, "max": 100},
        "k_smooth": {"type": "int", "default": 20, "min": 6, "max": 100},
        "reg_period": {"type": "int", "default": 64, "min": 6, "max": 100},
        "n1": {"type": "float", "default": 6.6, "min": 1.5, "max": 10.0},
        "ma_period": {"type": "int", "default": 29, "min": 6, "max": 100},
    },
}


def _positions(open_buy, open_short, buy0, short0, up_gate, dn_gate):
    n = len(buy0)
    pos = np.zeros(n, dtype=np.int8)
    p = 0
    for i in range(n):
        if p == 1 and short0[i]:
            p = 0
        elif p == -1 and buy0[i]:
            p = 0
        if p == 0:
            if open_buy[i] and up_gate[i]:
                p = 1
            elif open_short[i] and dn_gate[i]:
                p = -1
        pos[i] = p
    return pos


def indicators(df, params):
    """Các đường làm nên tín hiệu: f và s (hai đường cắt nhau) cùng EMA của nến liền trước.

    Dùng để vẽ biểu đồ đối chiếu (xem ke_plot.py); chính signals() cũng dùng đúng các đường này.
    """
    f = _kf(df, int(params["period"]), int(params["k_smooth"]), int(params["reg_period"]), float(params["n1"]))
    s = _wma(f, int(params["ma_period"]))
    ema = _ema(df["close"], int(params["ema_length"])).shift(1)
    return pd.DataFrame({"f": f, "s": s, "ema": ema}, index=df.index)


def signals(df, params):
    """Trả về Series 1 (mua) / -1 (bán khống) / 0 (đứng ngoài); giá trị tại nến i chỉ dùng thông tin tới nến i."""
    ind = indicators(df, params)
    f, s = ind["f"], ind["s"]
    buy0 = _cross(f, s)
    short0 = _cross(s, f)
    ema_prev = ind["ema"].to_numpy()
    close = df["close"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        up_gate = close > ema_prev
        dn_gate = close < ema_prev
    pos = _positions(_flip(buy0, short0), _flip(short0, buy0), buy0, short0, up_gate, dn_gate)
    return pd.Series(pos.astype(int), index=df.index)
