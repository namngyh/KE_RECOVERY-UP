"""Kiểm thử models.py: M0 phải trùng ke.signals; bộ lọc chỉ chặn lệnh mở mới; các biến thể chạy đúng."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import features, models  # noqa: E402
from strategies import ke_recovery as ke  # noqa: E402

PARAMS = {"ema_length": 40, "period": 8, "k_smooth": 4, "reg_period": 12, "n1": 6.6, "ma_period": 5}


def _lab(n_days=40, per_day=60, seed=1):
    rng = np.random.default_rng(seed)
    ts = []
    for d in pd.bdate_range("2021-01-04", periods=n_days):
        ts += [d + pd.Timedelta(hours=9) + pd.Timedelta(minutes=5 * k) for k in range(per_day)]
    n = len(ts)
    c = 1000 + np.cumsum(rng.normal(0, 1.0, n))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + rng.uniform(0, 0.5, n)
    l = np.minimum(o, c) - rng.uniform(0, 0.5, n)
    ms = (((pd.DatetimeIndex(ts) - pd.Timedelta(hours=7)) - pd.Timestamp("1970-01-01"))
          // pd.Timedelta(milliseconds=1)).astype("int64")
    raw = pd.DataFrame({"ts": ts, "open_time_ms": ms, "open": o, "high": h, "low": l, "close": c})
    vol = rng.integers(50, 150, n).astype(float)
    buy = np.floor(vol * rng.uniform(0.3, 0.7, n))
    bars = raw.assign(volume=vol, buy_vol=buy, sell_vol=vol - buy)
    F = features.build_features(bars)
    bc = pd.DataFrame({"near_expiry": np.zeros(n, bool), "after_holiday": np.zeros(n, bool),
                       "first_day_after_expiry": np.zeros(n, bool), "after_weekend": np.zeros(n, bool)})
    return models.Lab(raw, raw.copy(), F, bc), raw


def test_default_spec_equals_original_ke_signals():
    lab, raw = _lab()
    mine = models.build_signal(lab, {"params": PARAMS})
    ref = ke.signals(raw, PARAMS).to_numpy()
    assert (mine == ref).all() and (mine != 0).any()


def test_gate_blocks_only_new_entries_never_open_positions():
    lab, _ = _lab()
    base = models.build_signal(lab, {"params": PARAMS})
    N = 1200
    p = np.where(np.arange(lab.n) < N, 1.0, 0.0)                     # bộ lọc mở cho tới nến N, đóng sau đó
    lab.meta_p = (p, p)
    gated = models.build_signal(lab, {"params": PARAMS, "gates": [{"t": "meta", "tau": 0.5}]})
    assert (gated[:N] == base[:N]).all()
    run_start = np.flatnonzero((gated[1:] != 0) & (gated[1:] != gated[:-1])) + 1
    assert not (run_start >= N).any()                                  # không có lệnh mở mới sau N (kể cả đảo chiều)
    if base[N - 1] != 0:                                               # vị thế đang mở được giữ tới lúc M0 thoát
        change = np.flatnonzero(base[N:] != base[N - 1])
        stop = N + change[0] if len(change) else lab.n
        assert (gated[N:stop] == base[N - 1]).all()


def test_side_and_flat_eod_variants():
    lab, _ = _lab()
    assert (models.build_signal(lab, {"params": PARAMS, "side": "long"}) >= 0).all()
    assert (models.build_signal(lab, {"params": PARAMS, "side": "short"}) <= 0).all()
    eod = models.build_signal(lab, {"params": PARAMS, "flat_eod": True})
    last2 = lab.raw.groupby(lab.day).cumcount(ascending=False).to_numpy() < 2
    assert (eod[last2] == 0).all()


def test_added_feature_changes_signal_only_through_g():
    lab, _ = _lab()
    base = models.build_signal(lab, {"params": PARAMS})
    same = models.build_signal(lab, {"params": PARAMS, "add": [("I_tf", 0.0)]})
    assert (base == same).all()                                      # lambda = 0 phải trùng M0
    diff = models.build_signal(lab, {"params": PARAMS, "add": [("I_tf", 50.0)]})
    assert (base != diff).any()


def test_smoothers_and_m5b_and_adaptive_run():
    lab, _ = _lab()
    for spec in ({"smoother": "wma"}, {"smoother": "sma"}, {"g_mode": "m5b"}, {"g_mode": "rsi_only"},
                 {"g_mode": "k_only"}, {"ema_gate": False}, {"adaptive": True}):
        s = models.build_signal(lab, {"params": PARAMS, **spec})
        assert len(s) == lab.n and set(np.unique(s)) <= {-1, 0, 1}
    assert "M8_crosses_raw" in lab.diag


def test_unknown_options_raise():
    lab, _ = _lab()
    with pytest.raises(ValueError):
        models.build_signal(lab, {"g_mode": "xyz"})
    with pytest.raises(ValueError):
        models.build_signal(lab, {"smoother": "xyz"})
    with pytest.raises(ValueError):
        models.build_signal(lab, {"gates": [{"t": "unknown"}]})
