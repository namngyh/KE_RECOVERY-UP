"""Kiểm thử S1: runner và tái lập M0. Chạy: pytest colab/tests

Test hồi quy dùng file 5m thật nếu có (đường dẫn qua biến môi trường KE_5M_CSV, mặc định data/ ở gốc dự án);
không có file thì bỏ qua.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)
from ke_research import runner  # noqa: E402
from strategies import ke_recovery as ke  # noqa: E402

DEFAULT_5M = os.path.join(os.path.dirname(__file__), "..", "..", "data", "VN30F1M_5m_20171106-20261030.csv")
CSV_5M = os.environ.get("KE_5M_CSV", DEFAULT_5M)


def _toy_bars(n=60, start="2024-01-02 09:00"):
    ts = pd.date_range(start, periods=n, freq="5min")
    px = 1800 + np.arange(n) * 0.1
    ms = ((ts - pd.Timedelta(hours=7)) - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    return pd.DataFrame({"open_time_ms": ms.astype("int64"), "open": px, "high": px + 0.5,
                         "low": px - 0.5, "close": px + 0.1})


def test_config_matches_study_conventions():
    c = runner.make_config(tax_rate=0.001).contract
    assert (c.sizing, c.contracts, c.max_contracts) == ("fixed", 1, 1)
    assert c.multiplier == 100_000.0 and c.fee_rate == 0.00005 and c.slippage_points == 0.2
    assert c.initial_margin_rate == 0.1848 and c.force_sale == "none" and c.expiry == "close"
    assert c.tax_rate == 0.001


def test_signal_length_mismatch_raises():
    with pytest.raises(ValueError):
        runner.run(_toy_bars(), np.zeros(10))


def test_fill_at_next_open_with_slippage_and_one_contract():
    bars = _toy_bars()
    sig = np.zeros(len(bars), dtype=int)
    sig[5:20] = 1  # mua ở nến 5 -> khớp open nến 6; tín hiệu tắt ở nến 20 -> thoát khớp open nến 21
    res = runner.run(bars, sig, runner.make_config())
    tr = runner.trades_frame(res)
    assert len(tr) == 1 and tr.loc[0, "side"] == "long"
    assert (tr.loc[0, "entry_index"], tr.loc[0, "exit_index"]) == (6, 21)
    assert tr.loc[0, "entry_price"] == pytest.approx(bars.loc[6, "open"] + 0.2)
    assert tr.loc[0, "exit_price"] == pytest.approx(bars.loc[21, "open"] - 0.2)


def test_tax_reduces_net_pnl_by_expected_amount():
    bars = _toy_bars()
    sig = np.zeros(len(bars), dtype=int)
    sig[5:20] = 1
    a = runner.trades_frame(runner.run(bars, sig, runner.make_config(tax_rate=0.0))).loc[0]
    b = runner.trades_frame(runner.run(bars, sig, runner.make_config(tax_rate=0.001))).loc[0]
    expected = sum(0.001 * p * 100_000 * 0.1848 / 2 for p in (a["entry_price"], a["exit_price"]))
    assert (a["net_pnl"] - b["net_pnl"]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.skipif(not os.path.exists(CSV_5M), reason="không có file 5m")
def test_m0_replicates_original_report():
    df = pd.read_csv(CSV_5M)
    params = {k: v["default"] for k, v in ke.STRATEGY["params"].items()}
    res = runner.run(df, ke.signals(df, params).to_numpy(), runner.make_config(tax_rate=0.0))
    s = runner.summarize(res)
    assert s["trades"] == 1867 and s["wins"] == 890
    assert s["net_points"] == pytest.approx(4337.4, abs=0.05)
    assert s["final_equity_vnd"] / 1e6 == pytest.approx(533.74, abs=0.005)
