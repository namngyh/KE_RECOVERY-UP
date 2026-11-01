"""Kiểm thử predictive.py: lợi suất khả thi, kiểm định nhóm và Holm trên dữ liệu mô phỏng có hiệu ứng biết trước."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import models, predictive  # noqa: E402


def _tiny_lab():
    ts = [pd.Timestamp("2024-03-04 09:00") + pd.Timedelta(minutes=5 * k) for k in range(6)] + \
         [pd.Timestamp("2024-03-05 09:00") + pd.Timedelta(minutes=5 * k) for k in range(6)]
    o = np.arange(12, dtype=float) + 1000.0
    raw = pd.DataFrame({"ts": ts, "open_time_ms": np.arange(12), "open": o, "high": o, "low": o, "close": o})
    F = pd.DataFrame({"ER": np.zeros(12)})
    return models.Lab(raw, raw.copy(), F, None)


def test_eligible_returns_same_session_and_cost():
    lab = _tiny_lab()
    R, cost, ok = predictive.eligible_returns(lab, h=2)
    # nến 0: vào open[1]=1001, ra open[3]=1003 cùng phiên -> hợp lệ
    assert ok[0] and R[0] == pytest.approx((1003 - 1001) / 1001)
    assert cost[0] == pytest.approx(2 * 0.2 / 1001 + 2 * 0.00005)
    # nến 3,4,5: ra ở phiên sau -> không hợp lệ; nến 9..11 vượt cuối dữ liệu
    assert not ok[3] and not ok[4] and not ok[5] and not ok[9] and not ok[10] and not ok[11]
    assert ok[6] and R[6] == pytest.approx((1009 - 1007) / 1007)


def _panel(n_days=300, per_day=40, effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    day = np.repeat(pd.bdate_range("2019-01-02", periods=n_days), per_day)
    period = np.where(np.arange(n) < n // 2, "dev", "val")
    J = rng.normal(size=n)
    Y = effect * J + rng.normal(size=n)
    return pd.DataFrame({"day": day, "period": period, "q": 1.0, "Y": Y, "J": J, "A": rng.normal(size=n),
                         "dI_in": rng.normal(size=n), "dI_ov": rng.normal(size=n), "qdI_in": rng.normal(size=n)})


def test_group_tests_detect_true_effect_and_not_null():
    rows = predictive.group_tests(_panel(effect=0.3))
    h1 = {r["period"]: r for r in rows if r["test"] == "H1"}
    assert h1["dev"]["p"] < 0.001 and h1["val"]["p"] < 0.001 and h1["dev"]["diff"] > 0
    null = {r["test"]: r for r in predictive.group_tests(_panel(effect=0.0, seed=4)) if r["period"] == "val"}
    assert null["H2"]["p"] > 0.01 and null["H3_in"]["p"] > 0.01


def test_thresholds_come_from_dev_only():
    p = _panel(effect=0.0)
    lo, hi = predictive._thresholds(p, "J", 0.333)
    q = p[p["period"] == "dev"]["J"]
    assert lo == pytest.approx(q.quantile(0.333)) and hi == pytest.approx(q.quantile(0.667))
    p2 = p.copy()
    p2.loc[p2["period"] == "val", "J"] += 100                         # đổi val không được làm đổi ngưỡng
    assert predictive._thresholds(p2, "J", 0.333) == (lo, hi)


def test_calendar_test_flags_dependence():
    rng = np.random.default_rng(2)
    p = _panel(effect=0.0)
    p["near_expiry"] = (rng.random(len(p)) < 0.2).astype(float)
    p["after_holiday"] = 0.0
    p["first_day"] = 0.0
    p["after_weekend"] = 0.0
    p["Y"] = p["Y"] + 0.4 * p["near_expiry"]
    rows = {r["period"]: r for r in predictive.calendar_test(p)}
    assert rows["dev"]["p"] < 0.001 and rows["dev"]["betas"]["near_expiry"] > 0.2
