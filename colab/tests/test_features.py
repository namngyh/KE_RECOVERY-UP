"""Kiểm thử S3 - đặc trưng mua/bán (features.py) bằng dữ liệu mô phỏng nhỏ."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import features  # noqa: E402


def _bars(n_days=40, slots=("09:00", "09:05", "09:10"), seed=0):
    rng = np.random.default_rng(seed)
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:]))
          for d in pd.bdate_range("2024-01-02", periods=n_days) for s in slots]
    n = len(ts)
    vol = rng.integers(50, 150, n).astype(float)
    buy = np.floor(vol * rng.uniform(0.3, 0.7, n))
    close = 1000 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"ts": ts, "volume": vol, "buy_vol": buy, "sell_vol": vol - buy, "close": close})


def test_imbalance_formula_and_unclassified_flag():
    bars = _bars(3)
    bars.loc[0, ["volume", "buy_vol", "sell_vol"]] = [10, 6, 4]
    bars.loc[1, ["volume", "buy_vol", "sell_vol"]] = [50, 0, 0]            # có khối lượng nhưng không phân loại
    F = features.build_features(bars)
    assert F.loc[0, "I"] == pytest.approx(0.2)
    assert np.isnan(F.loc[1, "I"]) and bool(F.loc[1, "unclassified"]) and not bool(F.loc[0, "unclassified"])
    assert F["I"].dropna().between(-1, 1).all()


def test_activity_uses_only_prior_same_slot_volumes():
    bars = _bars(40)
    slot0 = bars["ts"].dt.hour * 60 + bars["ts"].dt.minute == 540
    bars.loc[slot0, "volume"] = 100.0
    last = bars.index[slot0][-1]
    bars.loc[last, "volume"] = 300.0
    F = features.build_features(bars)
    assert F.loc[last, "A"] == pytest.approx(np.log1p(300.0 / (100.0 + 1.0)))   # eps_V = 1


def test_delta_I_versions_differ_at_session_start():
    bars = _bars(5)
    F = features.build_features(bars)
    first_of_day = bars["ts"].dt.normalize() != bars["ts"].dt.normalize().shift()
    i = np.flatnonzero(first_of_day.to_numpy())[1]                          # đầu phiên thứ hai
    assert np.isnan(F.loc[i, "dI_in"]) and np.isfinite(F.loc[i, "dI_ov"])
    assert F.loc[i, "dI_ov"] == pytest.approx(F.loc[i, "I"] - F.loc[i - 1, "I"])


def test_features_do_not_use_future():
    bars = _bars(60)
    F1 = features.build_features(bars)
    b2 = bars.copy()
    cut = 100
    b2.loc[cut + 1:, ["volume", "buy_vol", "sell_vol"]] = b2.loc[cut + 1:, ["volume", "buy_vol", "sell_vol"]] * 4 + 20
    b2.loc[cut + 1:, "close"] = b2.loc[cut + 1:, "close"] * 2
    F2 = features.build_features(b2)
    cols = ["I", "A", "dI_in", "dI_ov", "I_t", "A_t", "dI_in_t", "dI_ov_t", "ER"]
    for c in cols:
        assert np.allclose(F1[c].iloc[:cut + 1].fillna(-9e9), F2[c].iloc[:cut + 1].fillna(-9e9)), c


def test_missing_standardized_filled_with_zero_and_exclude_atc_option():
    bars = _bars(40, slots=("09:00", "14:45"))
    bars.loc[bars["ts"].dt.hour == 14, ["buy_vol", "sell_vol"]] = 0           # ATC không phân loại
    F = features.build_features(bars)
    assert (F.loc[F["slot"] == 885, "I_tf"] == 0).all()                        # NaN -> 0 (quy tắc thiếu)
    Fx = features.build_features(bars, exclude_atc=True)
    assert Fx.loc[Fx["slot"] == 885, "A"].isna().all()                         # độ nhạy: ATC bị loại khỏi A


def test_efficiency_ratio_bounds():
    up = pd.Series(np.arange(50.0))
    assert features.efficiency_ratio(up, 10).dropna().eq(1.0).all()
    zig = pd.Series([0.0, 1.0] * 25)
    assert features.efficiency_ratio(zig, 10).dropna().max() < 0.2
