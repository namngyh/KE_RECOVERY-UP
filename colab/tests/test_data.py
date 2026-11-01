"""Kiểm thử S0 bằng dữ liệu tổng hợp nhỏ (không dùng dữ liệu thật). Chạy: pytest colab/tests"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import data  # noqa: E402

HEADER = "SYMBOL,TRADING_DATE,OPEN_PX,HIGH_PX,LOW_PX,CLOSE_PX,VOL,TRADING_TIME,BUY_VOL,BUY_VAL,SELL_VOL,SELL_VAL\n"
ROWS = [
    # 08:59 thuộc nến 5m nhãn 08:55; 09:00-09:04 thuộc nến 09:00; 09:05 thuộc nến 09:05
    "F,20240102,100.0,100.0,100.0,100.0,1,08:59:00,1,100,0,0",
    "F,20240102,100.0,101.0,99.5,100.5,10,09:00:00,6,600,4,400",
    "F,20240102,100.5,102.0,100.0,101.5,20,09:01:00,12,1200,8,800",
    "F,20240102,101.5,101.8,100.2,100.4,5,09:04:00,2,200,3,300",
    "F,20240102,100.4,100.9,100.1,100.6,7,09:05:00,3,300,4,400",
    "F,20240102,101.0,101.0,101.0,101.0,50,14:45:00,0,0,0,0",  # ATC: không phân loại
]


def _frame(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text(HEADER + "\n".join(ROWS) + "\n")
    return data.load_1m(str(p))


def test_open_time_ms_is_utc(tmp_path):
    df = _frame(tmp_path)
    # 2024-01-02 09:00 VN = 02:00 UTC = 1704160800000 ms
    row = df[df["ts"] == pd.Timestamp("2024-01-02 09:00")].iloc[0]
    assert row["open_time_ms"] == 1704160800000


def test_resample_buckets_and_aggregates(tmp_path):
    df = _frame(tmp_path)
    b = data.resample(df).set_index("ts")
    assert list(b.index.strftime("%H:%M")) == ["08:55", "09:00", "09:05", "14:45"]
    k = b.loc["2024-01-02 09:00"]
    assert (k["open"], k["high"], k["low"], k["close"]) == (100.0, 102.0, 99.5, 100.4)
    assert (k["volume"], k["buy_vol"], k["sell_vol"], k["n_minutes"]) == (35, 20, 15, 3)


def test_unclassified_flag_not_zero_filled(tmp_path):
    df = _frame(tmp_path)
    b = data.resample(df).set_index("ts")
    assert bool(b.loc["2024-01-02 14:45", "unclassified"]) is True
    assert bool(b.loc["2024-01-02 09:00", "unclassified"]) is False
    assert b.loc["2024-01-02 14:45", "volume"] == 50  # khối lượng vẫn giữ nguyên


def test_resample_conserves_totals(tmp_path):
    df = _frame(tmp_path)
    chk = data.check_resample(df, data.resample(df))
    assert all(v for k, v in chk.items() if k.endswith("_equal") or k == "minutes_conserved")


def test_audit_detects_problems(tmp_path):
    df = _frame(tmp_path)
    rep = data.audit_1m(df)
    assert rep["rows"] == 6 and rep["sessions"] == 1
    assert rep["unclassified_rows"] == 1
    assert rep["classified_gt_volume"] == 0
    assert rep["rows_before_0900"] == 1
    bad = df.copy()
    bad.loc[1, "high"] = 50.0  # high thấp hơn open/close
    assert data.audit_1m(bad)["high_below_open_or_close"] == 1
