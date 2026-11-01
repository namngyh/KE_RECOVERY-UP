"""Kiểm thử EGARCH-X (egarchx.py) trên dữ liệu mô phỏng có đáp án biết trước."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ke_research import egarchx as eg  # noqa: E402

Z = eg.ZGRID
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _moments(dist, nu=None, lam=None):
    p = np.exp(eg.logpdf(dist, Z, nu, lam))
    dz = Z[1] - Z[0]
    return float(_trapz(p, Z)), float(np.sum(Z * p) * dz), float(np.sum(Z ** 2 * p) * dz)


def test_densities_integrate_to_one_with_zero_mean_unit_variance():
    for dist, nu, lam in (("normal", None, None), ("student_t", 5.0, 0.0), ("skew_t", 6.0, -0.3), ("skew_t", 4.0, 0.4)):
        mass, mean, var = _moments(dist, nu, lam)
        assert mass == pytest.approx(1.0, abs=2e-3)
        assert mean == pytest.approx(0.0, abs=2e-3)
        assert var == pytest.approx(1.0, abs=2e-2 if nu and nu < 5 else 5e-3)


def test_expected_abs_closed_form_matches_numeric_and_differs_from_normal():
    nu = 5.0
    numeric = float(np.sum(np.abs(Z) * np.exp(eg.logpdf("student_t", Z, nu, 0.0))) * (Z[1] - Z[0]))
    assert eg.expected_abs("student_t", nu, 0.0) == pytest.approx(numeric, abs=1e-3)
    assert eg.expected_abs("student_t", nu, 0.0) < eg.expected_abs("normal")        # không mặc định E|z| của chuẩn
    assert eg.expected_abs("skew_t", nu, 0.0) == pytest.approx(eg.expected_abs("student_t", nu, 0.0), abs=2e-3)
    assert eg.expected_abs("normal") == pytest.approx(math.sqrt(2 / math.pi))


def test_cdf_grid_is_monotone_and_symmetric_for_student():
    g = eg.cdf_grid("student_t", np.array([math.log(6.0)]))
    assert np.all(np.diff(g) >= -1e-12) and g[0] == 0 and g[-1] == pytest.approx(1.0)
    assert np.interp(0.0, Z, g) == pytest.approx(0.5, abs=1e-3)


def _simulate(T, params, seed):
    rng = np.random.default_rng(seed)
    mu, omega, beta, kappa, theta, g1 = params
    x1 = rng.normal(0, 0.5, T)
    X = np.column_stack([np.concatenate([[0.0], x1[:-1]]), np.zeros(T), np.zeros((T, 3))])
    Ez = math.sqrt(2 / math.pi)
    r = np.empty(T); lh = omega / (1 - beta); z_prev = 0.0
    for t in range(T):
        if t > 0:
            lh = omega + beta * lh + kappa * (abs(z_prev) - Ez) + theta * z_prev + g1 * X[t, 0]
        z_prev = rng.normal()
        r[t] = mu + math.exp(0.5 * lh) * z_prev
    return r, X


def test_mle_recovers_true_parameters_and_pit_is_uniform():
    true = (0.02, 0.0, 0.9, 0.12, -0.05, 0.2)
    r, X = _simulate(4000, true, 11)
    lh0 = math.log(float(np.var(r[:250], ddof=1)))
    f = eg.fit("normal", r, X, lh0)
    base = f["theta"][: eg.N_BASE]
    assert abs(math.tanh(base[2]) - 0.9) < 0.07
    assert abs(base[3] - 0.12) < 0.07 and abs(base[5] - 0.2) < 0.08
    lh, z = eg.filter_paths(base, eg.expected_abs("normal"), r, X @ base[5:], lh0)
    from scipy import stats as sps
    u = sps.norm.cdf(z[250:])
    assert sps.kstest(u, "uniform").pvalue > 0.01
    assert np.std(z[250:], ddof=1) == pytest.approx(1.0, abs=0.06)


def test_forecast_never_uses_future_returns():
    r, X = _simulate(520, (0.0, 0.0, 0.9, 0.1, -0.05, 0.1), 5)
    a = eg.rolling_forecasts("normal", r, X, 250, 60)
    r2 = r.copy(); r2[400:] = r2[400:] * 7 + 3          # đổi toàn bộ dữ liệu từ phiên 400 trở đi
    b = eg.rolling_forecasts("normal", r2, X, 250, 60)
    assert np.allclose(a["h"].to_numpy()[250:401], b["h"].to_numpy()[250:401], rtol=1e-9)
    assert np.isnan(a["h"].to_numpy()[:250]).all()      # không dự báo trong cửa sổ huấn luyện đầu tiên


def test_evaluate_and_benchmarks_run():
    r, X = _simulate(700, (0.0, 0.0, 0.9, 0.1, -0.05, 0.1), 3)
    fc = eg.rolling_forecasts("student_t", r, X, 250, 80)
    rv = r ** 2
    mask = np.arange(len(r)) >= 250
    ev = eg.evaluate(fc, r, rv, mask)
    assert ev["n"] == 450 and 0 < ev["pit_ks_p"] <= 1 and "var01" in ev and "var05" in ev
    bm = eg.benchmark_variances(r, rv)
    assert np.isnan(bm["rolling_var_20"].iloc[19]) and np.isfinite(bm["rolling_var_20"].iloc[20])
    assert eg.qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(math.log(2.0) + 1.0)


def test_daily_inputs_excludes_lunch_gap_from_rv():
    # phiên giữa (02/01) có bước 101 -> 110 qua nghỉ trưa (11:30 -> 13:00) phải bị bỏ khỏi RV
    ts = pd.to_datetime(["2024-01-01 14:44", "2024-01-01 14:45",
                         "2024-01-02 11:29", "2024-01-02 11:30", "2024-01-02 13:00", "2024-01-02 13:01",
                         "2024-01-03 09:00", "2024-01-03 09:01"])
    close = [99.0, 99.5, 100.0, 101.0, 110.0, 111.0, 112.0, 113.0]
    bars = pd.DataFrame({"ts": ts, "close": close})
    days = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"], name="day")
    sess = pd.DataFrame({"r_cc": [np.nan, 0.01, 0.02], "r_night": [np.nan, 0.001, 0.003]}, index=days)
    cal = pd.DataFrame({"near_expiry": [False] * 3, "after_holiday": [False] * 3,
                        "first_day_after_expiry": [False] * 3}, index=days)
    d = eg.daily_inputs(bars, sess, cal)
    expected = (math.log(101 / 100) ** 2 + math.log(111 / 110) ** 2) * 1e4
    with_lunch = expected + math.log(110 / 101) ** 2 * 1e4
    assert pd.Timestamp("2024-01-01") not in d.index                 # phiên đầu không có r_night nên bị bỏ
    assert d.loc[pd.Timestamp("2024-01-02"), "rv_intra"] == pytest.approx(expected)
    assert d.loc[pd.Timestamp("2024-01-02"), "rv_intra"] < with_lunch
    assert d.loc[pd.Timestamp("2024-01-02"), "rv_proxy"] == pytest.approx(expected + (0.1) ** 2)
    X = eg.design_matrix(d)
    assert X.shape == (2, 5)
