"""Kiểm thử meta-labeling (meta.py): hồi quy logistic, AUC và quy tắc purging/embargo."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import meta  # noqa: E402


def test_fit_logistic_recovers_signal_and_predicts_probabilities():
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.normal(size=(n, 3))
    true_w = np.array([1.5, -1.0, 0.0])
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ true_w + 0.3)))).astype(float)
    m = meta.fit_logistic(X, y, C=10.0)
    w = m["w"] / m["sd"]                                     # trọng số trên thang gốc
    assert w[0] > 1.0 and w[1] < -0.6 and abs(w[2]) < 0.2
    p = meta.predict(m, X)
    assert 0 < p.min() and p.max() < 1 and meta.auc(y, p) > 0.75


def test_logistic_l2_shrinks_weights():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = (rng.random(300) < 0.5).astype(float)
    strong = meta.fit_logistic(X, y, C=0.001)
    weak = meta.fit_logistic(X, y, C=100.0)
    assert np.abs(strong["w"]).sum() < np.abs(weak["w"]).sum()


def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert meta.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert meta.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert np.isnan(meta.auc(np.ones(4), np.arange(4.0)))


def test_train_events_purges_unfinished_and_embargoed_trades():
    exits = pd.Series(pd.to_datetime(["2024-01-10", "2024-01-30", "2024-01-31", "2024-02-02"]))
    start = pd.Timestamp("2024-02-01")
    mask = meta.train_events(exits, start, embargo_days=1)
    assert list(mask) == [True, True, False, False]          # thoát 31/01 (trong embargo) và 02/02 (chưa thoát) bị loại
