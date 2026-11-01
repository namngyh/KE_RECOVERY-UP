"""S7 - EGARCH-X (đề cương mục 7). Chỉ dùng numpy/scipy.

    r_d = mu + sqrt(h_d) z_d
    log h_d = omega + beta log h_{d-1} + kappa (|z_{d-1}| - E|z|) + theta z_{d-1} + gamma' X_{d-1} + delta' D_d

* X_{d-1}: log RV_intra(d-1) và log(1e-3 + r_night(d-1)^2) (hai biến tách riêng để không đếm trùng, mục 7.2);
  D_d: cờ lịch đã biết trước phiên d (gần đáo hạn, sau kỳ nghỉ, ngày đầu sau đáo hạn).
* E|z| tính theo đúng phân phối giả định (mục 7.1): chuẩn, Student-t chuẩn hóa (nu > 2), skewed-t của Hansen (1994).
* Log-likelihood: l = sum [ -0.5 log h_d + log f_z((r_d - mu)/sqrt(h_d)) ]  (mục 7.3).
* Dự báo h^_{d|d-1} chỉ dùng thông tin đến cuối phiên d-1; tham số ước lượng lại theo cửa sổ mở rộng.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, special, stats as sps

from . import stats as st

_trapz = getattr(np, "trapezoid", None) or np.trapz   # NumPy >= 2.0 đổi tên trapz -> trapezoid

DISTS = ("normal", "student_t", "skew_t")
N_BASE = 10  # mu, omega, b(beta=tanh b), kappa, theta, gamma1, gamma2, delta1..3
N_SHAPE = {"normal": 0, "student_t": 1, "skew_t": 2}
ZGRID = np.linspace(-40.0, 40.0, 16001)


# ------------------------------------------------------------------ phân phối chuẩn hóa (mean 0, var 1)
def _shape(dist: str, s: np.ndarray):
    if dist == "normal":
        return None, None
    nu = 2.05 + math.exp(float(np.clip(s[0], -8, 8)))
    lam = math.tanh(float(s[1])) if dist == "skew_t" else 0.0
    return nu, lam


def _hansen_abc(nu: float, lam: float):
    c = math.exp(special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)) / math.sqrt(math.pi * (nu - 2))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = math.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


def logpdf(dist: str, z, nu=None, lam=None):
    z = np.asarray(z, float)
    if dist == "normal":
        return -0.5 * math.log(2 * math.pi) - 0.5 * z * z
    if dist == "student_t":
        return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
                - (nu + 1) / 2 * np.log1p(z * z / (nu - 2)))
    a, b, c = _hansen_abc(nu, lam)
    left = z < -a / b
    denom = np.where(left, 1 - lam, 1 + lam)
    u = (b * z + a) / denom
    return math.log(b) + math.log(c) - (nu + 1) / 2 * np.log1p(u * u / (nu - 2))


def expected_abs(dist: str, nu=None, lam=None) -> float:
    """E|z| theo phân phối đã chọn (không mặc định chuẩn khi dùng Student-t/skewed-t)."""
    if dist == "normal":
        return math.sqrt(2 / math.pi)
    if dist == "student_t":
        return 2 * math.sqrt(nu - 2) * math.exp(special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)) \
            / (math.sqrt(math.pi) * (nu - 1))
    pdf = np.exp(logpdf(dist, ZGRID, nu, lam))
    return float(_trapz(np.abs(ZGRID) * pdf, ZGRID))


def cdf_grid(dist: str, shape: np.ndarray):
    nu, lam = _shape(dist, shape)
    pdf = np.exp(logpdf(dist, ZGRID, nu, lam))
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(ZGRID))])
    return cdf / cdf[-1]


# ------------------------------------------------------------------ bộ lọc và log-likelihood
def _unpack(dist, theta):
    return theta[:N_BASE], theta[N_BASE:N_BASE + N_SHAPE[dist]]


def filter_paths(base, Ez: float, r: np.ndarray, u: np.ndarray, lh0: float):
    mu, omega, b, kappa, theta_ = base[0], base[1], math.tanh(base[2]), base[3], base[4]
    T = len(r)
    lh = np.empty(T); z = np.empty(T)
    prev_lh, prev_z = lh0, None
    for t in range(T):
        cur = omega + b * prev_lh + u[t]
        if prev_z is not None:
            cur += kappa * (abs(prev_z) - Ez) + theta_ * prev_z
        cur = -30.0 if cur < -30.0 else 30.0 if cur > 30.0 else cur
        lh[t] = cur
        zt = (r[t] - mu) * math.exp(-0.5 * cur)
        z[t] = zt
        prev_lh, prev_z = cur, zt
    return lh, z


def neg_loglik(theta, dist, r, X, lh0):
    base, shape = _unpack(dist, theta)
    nu, lam = _shape(dist, shape)
    Ez = expected_abs(dist, nu, lam)
    u = X @ base[5:N_BASE]
    lh, z = filter_paths(base, Ez, r, u, lh0)
    ll = -0.5 * lh[1:] + logpdf(dist, z[1:], nu, lam)
    val = -float(np.sum(ll))
    return val if np.isfinite(val) else 1e12


def _x0(dist, r, lh0):
    base = [float(np.mean(r)), 0.1 * lh0, 1.47, 0.1, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    shape = {"normal": [], "student_t": [math.log(6.0)], "skew_t": [math.log(6.0), 0.0]}[dist]
    return np.array(base + shape, float)


def _bounds(dist):
    b = [(None, None), (-10, 10), (-4, 4), (-2, 2), (-2, 2)] + [(-3, 3)] * 5
    return b + {"normal": [], "student_t": [(-3, 5)], "skew_t": [(-3, 5), (-3, 3)]}[dist]


def fit(dist: str, r: np.ndarray, X: np.ndarray, lh0: float, x0=None, maxiter: int = 120) -> dict:
    x0 = _x0(dist, r, lh0) if x0 is None else x0
    res = optimize.minimize(neg_loglik, x0, args=(dist, r, X, lh0), method="L-BFGS-B", bounds=_bounds(dist),
                            options={"maxiter": maxiter, "maxfun": 60 * len(x0) * 4})
    return {"theta": res.x, "nll": float(res.fun), "converged": bool(res.success), "n": len(r)}


# ------------------------------------------------------------------ dự báo ngoài mẫu
def rolling_forecasts(dist: str, r: np.ndarray, X: np.ndarray, min_train: int, refit_every: int,
                      alphas=(0.01, 0.05)) -> pd.DataFrame:
    """Dự báo 1 bước h^_{t|t-1} cho t >= min_train, ước lượng lại mỗi `refit_every` phiên bằng dữ liệu TRƯỚC t0."""
    T = len(r)
    lh0 = math.log(float(np.var(r[:min_train], ddof=1)))
    cols = {k: np.full(T, np.nan) for k in ("h", "z", "u", "ls") + tuple(f"q{int(round(a * 100)):02d}" for a in alphas)}
    theta = None
    fits = []
    for t0 in range(min_train, T, refit_every):
        t1 = min(t0 + refit_every, T)
        f = fit(dist, r[:t0], X[:t0], lh0, x0=theta, maxiter=200 if theta is None else 80)
        theta = f["theta"]
        fits.append({"t0": t0, "nll": f["nll"], "converged": f["converged"]})
        base, shape = _unpack(dist, theta)
        nu, lam = _shape(dist, shape)
        Ez = expected_abs(dist, nu, lam)
        lh, z = filter_paths(base, Ez, r[:t1], X[:t1] @ base[5:N_BASE], lh0)
        grid = cdf_grid(dist, shape)
        sl = slice(t0, t1)
        zz = z[sl]
        cols["h"][sl] = np.exp(lh[sl])
        cols["z"][sl] = zz
        cols["u"][sl] = np.interp(zz, ZGRID, grid)
        cols["ls"][sl] = 0.5 * lh[sl] - logpdf(dist, zz, nu, lam)
        for a in alphas:
            cols[f"q{int(round(a * 100)):02d}"][sl] = base[0] + np.exp(0.5 * lh[sl]) * np.interp(a, grid, ZGRID)
    out = pd.DataFrame(cols)
    out.attrs["fits"] = fits
    return out


# ------------------------------------------------------------------ đối chứng đơn giản và đánh giá
def benchmark_variances(r: np.ndarray, rv_proxy: np.ndarray) -> pd.DataFrame:
    T = len(r)
    roll = np.full(T, np.nan); ew = np.full(T, np.nan); rv5 = np.full(T, np.nan)
    h = float(np.var(r[:20], ddof=1))
    for t in range(T):
        if t >= 20:
            roll[t] = float(np.var(r[t - 20:t], ddof=1))
            h = 0.94 * h + 0.06 * r[t - 1] ** 2
            ew[t] = h
        if t >= 5:
            rv5[t] = float(np.mean(rv_proxy[t - 5:t]))
    return pd.DataFrame({"rolling_var_20": roll, "ewma_0.94": ew, "mean_RV_5": rv5})


def qlike(h: np.ndarray, rv: np.ndarray) -> np.ndarray:
    return np.log(h) + rv / h


def evaluate(fc: pd.DataFrame, r: np.ndarray, rv_proxy: np.ndarray, mask: np.ndarray, alphas=(0.01, 0.05)) -> dict:
    """Đánh giá dự báo trên các phiên có mask=True: QLIKE, log-score, PIT, VaR, Ljung-Box (mục 7.4)."""
    m = mask & np.isfinite(fc["h"].to_numpy())
    h = fc["h"].to_numpy()[m]
    out = {"n": int(m.sum()), "qlike": float(np.mean(qlike(h, rv_proxy[m]))),
           "log_score": float(fc["ls"].to_numpy()[m].mean())}
    u = fc["u"].to_numpy()[m]
    out["pit_ks_p"] = float(sps.kstest(u, "uniform").pvalue)
    z = fc["z"].to_numpy()[m]
    out["ljung_box_z_p"] = st.ljung_box(z, 10)["p"]
    out["ljung_box_z2_p"] = st.ljung_box(z ** 2, 10)["p"]
    for a in alphas:
        key = f"q{int(round(a * 100)):02d}"
        q = fc[key].to_numpy()[m]
        v = r[m] < q
        out[f"var{int(round(a * 100)):02d}"] = {
            "rate": float(v.mean()), "kupiec_p": st.kupiec(v, a)["p"],
            "christoffersen_p": st.christoffersen_independence(v)["p"],
            "pinball": float(np.mean(np.where(r[m] < q, (a - 1) * (r[m] - q), a * (r[m] - q))))}
    return out


# ------------------------------------------------------------------ dữ liệu ngày cho EGARCH-X
def daily_inputs(bars_1m: pd.DataFrame, session_returns: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Bảng theo phiên: r (%, r_cc đã hiệu chỉnh roll), rv_intra (%^2), r_night (%), cờ lịch.

    RV_intra = tổng bình phương lợi suất 1 phút trong phiên, bỏ bước qua nghỉ trưa (khoảng cách > 60 phút giữa hai
    nến liên tiếp). Trong một phiên không có roll nên giá thực dùng được (không cần hiệu chỉnh)."""
    day = bars_1m["ts"].dt.normalize()
    lr = np.log(bars_1m["close"]).diff()
    gap = bars_1m["ts"].diff().dt.total_seconds().div(60)
    ok = (day == day.shift()) & (gap <= 60)
    rv = (lr.where(ok) ** 2).groupby(day).sum() * 1e4
    df = pd.DataFrame({"r": session_returns["r_cc"] * 100.0, "r_night": session_returns["r_night"] * 100.0,
                       "rv_intra": rv})
    df = df.join(calendar[["near_expiry", "after_holiday", "first_day_after_expiry"]])
    df["rv_proxy"] = df["rv_intra"] + df["r_night"] ** 2
    return df.iloc[1:].copy()           # phiên đầu không có r_night


def design_matrix(d: pd.DataFrame) -> np.ndarray:
    """X_t = [log RV_{t-1}, log(1e-3 + r_night_{t-1}^2), D_t]; hàng đầu dùng chính phiên đầu làm 'trước' (không vào log-lik)."""
    x1 = np.log(d["rv_intra"].to_numpy() + 1e-8)
    x2 = np.log(1e-3 + d["r_night"].to_numpy() ** 2)
    x1p = np.concatenate([[x1[0]], x1[:-1]]); x2p = np.concatenate([[x2[0]], x2[:-1]])
    D = d[["near_expiry", "after_holiday", "first_day_after_expiry"]].to_numpy(float)
    return np.column_stack([x1p, x2p, D])
