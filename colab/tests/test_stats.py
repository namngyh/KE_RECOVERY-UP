"""Kiểm thử công cụ thống kê (stats.py) bằng dữ liệu mô phỏng có đáp án biết trước."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import stats as st  # noqa: E402


def test_cluster_ols_recovers_beta_and_matches_lstsq():
    rng = np.random.default_rng(0)
    n = 4000
    g = np.repeat(np.arange(200), 20)
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    r = st.cluster_ols(y, X, g)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(r["beta"], ref)
    assert abs(r["beta"][1] - 2.0) < 0.1 and r["p"][1] < 1e-6 and r["clusters"] == 200


def test_cluster_se_larger_than_naive_when_errors_clustered():
    rng = np.random.default_rng(1)
    G, m = 100, 30
    g = np.repeat(np.arange(G), m)
    shock = np.repeat(rng.normal(size=G), m)           # sai số chung trong cụm
    x = np.repeat(rng.normal(size=G), m)               # biến chỉ thay đổi theo cụm
    y = shock + rng.normal(scale=0.3, size=G * m)
    X = np.column_stack([np.ones(G * m), x])
    clustered = st.cluster_ols(y, X, g)["se"][1]
    naive = st.cluster_ols(y, X, np.arange(G * m))["se"][1]
    assert clustered > 2 * naive


def test_mean_diff_and_wald():
    rng = np.random.default_rng(2)
    n = 6000
    g = np.repeat(np.arange(300), 20)
    hi = rng.random(n) < 0.4
    lo = ~hi
    y = 0.5 * hi + rng.normal(size=n)
    r = st.mean_diff(y, hi, lo, g, one_sided=True)
    assert abs(r["diff"] - 0.5) < 0.15 and r["p"] < 0.001
    X = np.column_stack([np.ones(n), hi.astype(float), rng.normal(size=n)])
    fit = st.cluster_ols(y, X, g)
    w = st.wald_test(fit["beta"], fit["cov"], [1, 2])
    assert w["df"] == 2 and w["p"] < 0.001


def test_holm_known_example():
    assert np.allclose(st.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(st.holm([0.5]), [0.5])


def test_sharpe_and_mdd():
    d = np.array([1.0, -2.0, 3.0, -1.0])
    assert st.max_drawdown(d) == pytest.approx(2.0)
    assert st.sharpe(d, 250) == pytest.approx(np.sqrt(250) * d.mean() / d.std(ddof=1))
    assert st.max_drawdown(np.array([1.0, 1.0, 1.0])) == 0.0


def test_paired_bootstrap_detects_improvement_and_not_noise():
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, 600)
    better = base + 0.35 + rng.normal(0, 0.2, 600)
    assert st.paired_block_bootstrap(better, base, 5, 1000, 1)["p_one_sided"] < 0.05
    # dưới H0 (kỳ vọng bằng nhau) tỷ lệ p<0,10 phải xấp xỉ 10%: kiểm tra hiệu chỉnh trên nhiều mô phỏng
    ps = []
    for s in range(120):
        g = np.random.default_rng(1000 + s)
        b = g.normal(0, 1, 400)
        ps.append(st.paired_block_bootstrap(b + g.normal(0, 0.3, 400), b, 5, 300, s)["p_one_sided"])
    assert 0.02 < float(np.mean(np.array(ps) < 0.10)) < 0.22
    with pytest.raises(ValueError):
        st.paired_block_bootstrap(base[:10], base[:11])


def test_spa_rejects_when_a_model_is_better_only():
    rng = np.random.default_rng(4)
    T, K = 600, 8
    null = rng.normal(0, 1, (T, K))
    assert st.spa_test(null, 5, 1000, 1)["p_value"] > 0.1
    good = null.copy()
    good[:, 3] += 0.4
    r = st.spa_test(good, 5, 1000, 1)
    assert r["p_value"] < 0.05 and r["best_k"] == 3


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1, (800, 30))
    assert 0.25 < st.pbo_cscv(noise, 16)["pbo"] < 0.75
    edge = rng.normal(0, 1, (800, 30))
    edge[:, 7] += 0.25                                 # cấu hình có lợi thế thật, ổn định
    assert st.pbo_cscv(edge, 16)["pbo"] < 0.2


def test_deflated_sharpe_decreases_with_more_trials():
    few = st.deflated_sharpe(0.12, np.random.default_rng(6).normal(0.0, 0.05, 5), 1000, 0.0, 3.0)
    many = st.deflated_sharpe(0.12, np.random.default_rng(6).normal(0.0, 0.05, 500), 1000, 0.0, 3.0)
    assert many["sr_star"] > few["sr_star"] and many["dsr"] < few["dsr"]
    assert st.deflated_sharpe(0.12, [0.12], 1000, 0.0, 3.0)["sr_star"] == 0.0


def test_kupiec_and_christoffersen():
    v = np.zeros(1000, bool); v[:10] = True            # đúng 1% vi phạm
    assert st.kupiec(v, 0.01)["p"] > 0.9
    v2 = np.zeros(1000, bool); v2[:60] = True          # 6% vi phạm so với 1% kỳ vọng
    assert st.kupiec(v2, 0.01)["p"] < 0.001
    rng = np.random.default_rng(7)
    iid = rng.random(5000) < 0.05
    assert st.christoffersen_independence(iid)["p"] > 0.01
    clustered = np.zeros(5000, bool)
    for s in range(0, 5000, 100):
        clustered[s:s + 5] = True                      # vi phạm dồn cụm
    assert st.christoffersen_independence(clustered)["p"] < 0.001


def test_ljung_box_white_noise_vs_ar1():
    rng = np.random.default_rng(8)
    wn = rng.normal(size=2000)
    assert st.ljung_box(wn, 10)["p"] > 0.01
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.5 * ar[i - 1] + rng.normal()
    assert st.ljung_box(ar, 10)["p"] < 1e-10


def test_dm_test_prefers_lower_loss():
    rng = np.random.default_rng(9)
    a = rng.normal(1.0, 0.3, 800)
    b = a + 0.2 + rng.normal(0, 0.3, 800)
    assert st.dm_test(a, b)["p_a_better"] < 0.01
    assert st.dm_test(b, a)["p_a_better"] > 0.99
