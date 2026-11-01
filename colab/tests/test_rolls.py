"""Kiểm thử S2 - roll và chuỗi hiệu chỉnh (đề cương mục 5.2, 5.3). Dữ liệu tổng hợp."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import rolls  # noqa: E402
from strategies import ke_recovery as ke  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01")


def _bars(ts_list, o, h, l, c):
    ts = pd.DatetimeIndex(ts_list)
    ms = (((ts - pd.Timedelta(hours=7)) - EPOCH) // pd.Timedelta(milliseconds=1)).astype("int64")
    return pd.DataFrame({"ts": ts, "open_time_ms": ms, "open": o, "high": h, "low": l, "close": c})


def test_cumulative_offset_steps_at_roll_bar():
    cum = rolls.cumulative_offset(10, [3, 7], [5.0, -2.0])
    assert list(cum) == [0, 0, 0, 5, 5, 5, 5, 3, 3, 3]


def test_roll_table_delta_formula_and_causality_of_dates():
    # 18/01/2024 là thứ Năm thứ ba của tháng 1 (ngày đáo hạn); 19/01 là phiên đầu của hợp đồng mới
    days = ["2024-01-17", "2024-01-18", "2024-01-19", "2024-01-22", "2024-01-23"]
    ts, o, c = [], [], []
    px = {"2024-01-17": 1000.0, "2024-01-18": 1001.0, "2024-01-19": 1012.0, "2024-01-22": 1013.0, "2024-01-23": 1014.0}
    for d in days:
        for hhmm in ("09:00", "14:45"):
            ts.append(pd.Timestamp(f"{d} {hhmm}")); o.append(px[d]); c.append(px[d])
    bars = _bars(ts, o, o, o, c)
    index_close = pd.Series({pd.Timestamp(d): v for d, v in
                             zip(days, [1000.0, 1000.5, 1003.0, 1004.0, 1005.0])})
    rt = rolls.roll_table(bars, index_close)
    assert len(rt) == 1
    r = rt.iloc[0]
    assert (r["exp_day"], r["new_day"]) == (pd.Timestamp("2024-01-18"), pd.Timestamp("2024-01-19"))
    assert r["roll_idx"] == 4  # nến đầu của 19/01
    assert r["basis_exp"] == pytest.approx(1001.0 - 1000.5) and r["basis_next"] == pytest.approx(1012.0 - 1003.0)
    assert r["delta"] == pytest.approx((1012.0 - 1003.0) - (1001.0 - 1000.5))


def test_forward_adjust_removes_roll_jump_and_keeps_raw():
    n = 30
    rng = np.random.default_rng(1)
    base = 1000 + np.cumsum(rng.normal(0, 1, n))
    raw = base.copy(); raw[10:] += 7.0  # bước nhảy do roll
    bars = _bars(pd.date_range("2024-01-02 09:00", periods=n, freq="5min"), raw, raw + 1, raw - 1, raw)
    rt = pd.DataFrame({"roll_idx": [10], "delta": [7.0]})
    adj = rolls.forward_adjust(bars, rt)
    assert np.allclose(adj["close"], base)                       # chuỗi liền mạch
    assert np.allclose(adj["raw_close"], raw)                    # giá thực được giữ nguyên
    assert np.allclose(adj["cum_offset"][:10], 0) and np.allclose(adj["cum_offset"][10:], 7.0)


def test_forward_adjust_is_causal_and_equivalent_to_asof_back_adjusted():
    """Chỉ báo KE trên chuỗi thuận phải cho cùng f, s và cổng EMA như chuỗi cộng lùi nhân quả (đề cương)."""
    n = 400
    rng = np.random.default_rng(7)
    px = 1000 + np.cumsum(rng.normal(0, 1.0, n))
    deltas = {100: 5.0, 200: -3.0, 300: 8.0}
    raw = px.copy()
    for i, d in deltas.items():
        raw[i:] += d
    bars = _bars(pd.date_range("2024-01-02 09:00", periods=n, freq="5min"), raw, raw + 0.5, raw - 0.5, raw + 0.1)
    rt = pd.DataFrame({"roll_idx": list(deltas), "delta": list(deltas.values())})
    params = {"ema_length": 20, "period": 5, "k_smooth": 3, "reg_period": 8, "n1": 6.6, "ma_period": 4}
    adj = rolls.forward_adjust(bars, rt)
    ind_f = ke.indicators(adj, params)
    for t in (150, 199, 200, 205, 299, 301, 399):
        pre = rolls.asof_back_adjusted(bars, rt, t)
        ind_a = ke.indicators(pre, params)
        assert ind_f["f"].iloc[t] == pytest.approx(ind_a["f"].iloc[t], abs=1e-9)
        assert ind_f["s"].iloc[t] == pytest.approx(ind_a["s"].iloc[t], abs=1e-9)
        assert (adj["close"].iloc[t] > ind_f["ema"].iloc[t]) == (pre["close"].iloc[t] > ind_a["ema"].iloc[t])


def test_future_roll_does_not_change_past_adjusted_prices():
    n = 50
    px = 1000 + np.arange(n, dtype=float)
    bars = _bars(pd.date_range("2024-01-02 09:00", periods=n, freq="5min"), px, px, px, px)
    a = rolls.forward_adjust(bars, pd.DataFrame({"roll_idx": [20], "delta": [4.0]}))
    b = rolls.forward_adjust(bars, pd.DataFrame({"roll_idx": [20, 40], "delta": [4.0, 9.0]}))
    assert np.allclose(a["close"].iloc[:40], b["close"].iloc[:40])  # roll ở 40 không đổi quá khứ


def test_intraday_overnight_returns_decomposition():
    ts = [pd.Timestamp("2024-01-02 09:00"), pd.Timestamp("2024-01-02 14:45"),
          pd.Timestamp("2024-01-03 09:00"), pd.Timestamp("2024-01-03 14:45")]
    o = [100.0, 101.0, 103.0, 104.0]
    c = [100.5, 102.0, 103.5, 105.0]
    bars = _bars(ts, o, o, o, c)
    adj = rolls.forward_adjust(bars, pd.DataFrame({"roll_idx": [], "delta": []}))
    r = rolls.intraday_overnight_returns(adj)
    d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
    assert r.loc[d1, "r_intra"] == pytest.approx(np.log(102.0 / 100.0))
    assert np.isnan(r.loc[d1, "r_night"])
    assert r.loc[d2, "r_night"] == pytest.approx(np.log(103.0 / 102.0))
    assert r.loc[d2, "r_cc"] == pytest.approx(r.loc[d2, "r_night"] + r.loc[d2, "r_intra"])
