"""Công cụ thống kê chỉ dùng numpy/pandas/scipy (đề cương mục 10-11).

* cluster_ols / wald_test / mean_diff   : hồi quy với sai số chuẩn theo cụm (phiên), mục 10.2-10.3
* holm                                  : hiệu chỉnh đa kiểm định
* sharpe / max_drawdown                 : chỉ số U của đề cương mục 11.3
* paired_block_bootstrap                : so sánh theo cặp theo khối phiên, mục 11.4
* spa_test                              : Hansen (2005) SPA, bootstrap theo khối, mục 11.4
* pbo_cscv                              : xác suất backtest quá khớp, mục 11.5
* deflated_sharpe                       : DSR, mục 11.6
* kupiec / christoffersen / ljung_box / dm_test : đánh giá EGARCH-X, mục 7.4
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------- hồi quy theo cụm
def cluster_ols(y, X, groups) -> dict:
    """OLS với ma trận hiệp phương sai cụm (CR1). X phải gồm hệ số chặn nếu cần. Trả về beta, se, t, p, cov, r2."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    g = np.unique(np.asarray(groups), return_inverse=True)[1]
    n, k = X.shape
    G = int(g.max()) + 1
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    e = y - X @ beta
    scores = np.zeros((G, k))
    np.add.at(scores, g, X * e[:, None])
    cov = xtx_inv @ (scores.T @ scores) @ xtx_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=max(G - 1, 1))
    tss = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": t, "p": p, "cov": cov, "n": n, "clusters": G,
            "r2": 1 - float((e ** 2).sum()) / tss if tss > 0 else float("nan")}


def wald_test(beta, cov, idx) -> dict:
    """H0: beta[idx] = 0 (Wald, chi-bình phương)."""
    idx = list(idx)
    b = np.asarray(beta)[idx]
    V = np.asarray(cov)[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.pinv(V) @ b)
    return {"stat": stat, "df": len(idx), "p": float(stats.chi2.sf(stat, len(idx)))}


def mean_diff(y, hi, lo, groups, one_sided: bool = False) -> dict:
    """E[y|hi] - E[y|lo] bằng hồi quy y ~ 1 + 1(hi) trên hi|lo với sai số chuẩn theo cụm."""
    y = np.asarray(y, float); hi = np.asarray(hi, bool); lo = np.asarray(lo, bool); groups = np.asarray(groups)
    m = (hi | lo) & np.isfinite(y)
    X = np.column_stack([np.ones(m.sum()), hi[m].astype(float)])
    r = cluster_ols(y[m], X, groups[m])
    t = float(r["t"][1])
    df = max(r["clusters"] - 1, 1)
    return {"diff": float(r["beta"][1]), "se": float(r["se"][1]), "t": t,
            "p": float(stats.t.sf(t, df)) if one_sided else float(r["p"][1]),
            "n_hi": int((hi & m).sum()), "n_lo": int((lo & m).sum()), "clusters": r["clusters"]}


def holm(pvals) -> np.ndarray:
    """p đã hiệu chỉnh Holm-Bonferroni, giữ nguyên thứ tự đầu vào."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- chỉ số hiệu quả
def sharpe(daily, periods_per_year: int = 250, axis: int = -1):
    """SR thường niên hóa = sqrt(N_y) * mean / std (rf = 0, lợi nhuận tính theo điểm P&L ròng; mục 11.3)."""
    a = np.asarray(daily, dtype=float)
    sd = a.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(periods_per_year) * a.mean(axis=axis) / sd


def max_drawdown(daily) -> float:
    """MDD theo điểm của đường lợi nhuận tích lũy."""
    cum = np.cumsum(np.asarray(daily, dtype=float))
    if not len(cum):
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float((peak - cum).max())


# ---------------------------------------------------------------- bootstrap theo khối
def _block_index(T: int, block: int, reps: int, rng) -> np.ndarray:
    nb = math.ceil(T / block)
    starts = rng.integers(0, max(T - block, 0) + 1, size=(reps, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :T]


def paired_block_bootstrap(a, b, block: int = 5, reps: int = 2000, seed: int = 0,
                           periods_per_year: int = 250) -> dict:
    """Kiểm định H0: U(a) - U(b) <= 0 (U = Sharpe), bootstrap theo khối phiên, cùng chỉ số mẫu cho cả hai chuỗi."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("Hai chuỗi phải cùng độ dài (cùng các phiên)")
    rng = np.random.default_rng(seed)
    idx = _block_index(len(a), block, reps, rng)
    d_star = sharpe(a[idx], periods_per_year, axis=1) - sharpe(b[idx], periods_per_year, axis=1)
    d_hat = float(sharpe(a, periods_per_year) - sharpe(b, periods_per_year))
    p = float((1 + np.sum((d_star - d_hat) >= d_hat)) / (reps + 1))
    lo, hi = np.nanpercentile(d_star, [2.5, 97.5])
    return {"dU": d_hat, "p_one_sided": p, "ci95": [float(lo), float(hi)],
            "U_a": float(sharpe(a, periods_per_year)), "U_b": float(sharpe(b, periods_per_year))}


def spa_test(diffs, block: int = 5, reps: int = 2000, seed: int = 0) -> dict:
    """SPA của Hansen (2005). `diffs` shape (T, K): chênh lệch P&L ngày của K mô hình so với chuẩn (lớn hơn = tốt hơn).

    H0: max_k E[d_k] <= 0. Thống kê T = max(0, max_k sqrt(T) dbar_k / omega_k), omega_k là sai số chuẩn bootstrap.
    Căn giữa nhất quán: mô hình tệ rõ rệt (t_k < -sqrt(2 log log T)) giữ nguyên trung bình âm, các mô hình còn lại
    được căn về 0; p-value lấy từ phân phối bootstrap theo khối."""
    d = np.asarray(diffs, float)
    T, K = d.shape
    rng = np.random.default_rng(seed)
    idx = _block_index(T, block, reps, rng)
    dbar = d.mean(axis=0)
    boot = d[idx].mean(axis=1)                                   # (reps, K)
    omega = np.sqrt(T) * boot.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    t_k = np.sqrt(T) * dbar / omega
    t_obs = float(max(0.0, t_k.max()))
    thr = math.sqrt(2.0 * math.log(math.log(max(T, 3))))
    g = np.where(t_k >= -thr, dbar, 0.0)
    t_star = np.maximum(0.0, (np.sqrt(T) * (boot - g) / omega).max(axis=1))
    return {"t_stat": t_obs, "p_value": float((1 + np.sum(t_star >= t_obs)) / (reps + 1)), "K": K, "T": T,
            "best_k": int(np.argmax(dbar))}


# ---------------------------------------------------------------- PBO và DSR
def pbo_cscv(M, blocks: int = 16, periods_per_year: int = 250) -> dict:
    """Combinatorially Symmetric Cross-Validation. M shape (T, N): P&L ngày của N cấu hình đã thử."""
    M = np.asarray(M, float)
    T, N = M.shape
    if blocks % 2:
        raise ValueError("blocks phải chẵn")
    L = T // blocks
    B = M[: L * blocks].reshape(blocks, L, N)
    s1, s2 = B.sum(axis=1), (B ** 2).sum(axis=1)                 # (blocks, N)
    combos = np.array(list(itertools.combinations(range(blocks), blocks // 2)))
    mask = np.zeros((len(combos), blocks))
    np.put_along_axis(mask, combos, 1.0, axis=1)
    inv = 1.0 - mask

    def sr(m):
        n = m.sum(axis=1, keepdims=True) * L
        mean = (m @ s1) / n
        var = ((m @ s2) - n * mean ** 2) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(periods_per_year) * mean / np.sqrt(np.where(var > 0, var, np.nan))

    is_sr, oos_sr = sr(mask), sr(inv)
    best = np.argmax(np.where(np.isnan(is_sr), -np.inf, is_sr), axis=1)
    oos_best = oos_sr[np.arange(len(combos)), best]
    rank = (oos_sr < oos_best[:, None]).sum(axis=1) + 1          # 1..N (cao hơn = tốt hơn)
    omega = rank / (N + 1.0)
    lam = np.log(omega / (1 - omega))
    return {"pbo": float(np.mean(lam < 0)), "combos": int(len(combos)), "N": int(N),
            "median_logit": float(np.median(lam))}


def deflated_sharpe(sr_hat: float, sr_trials, T: int, skew: float, kurt: float) -> dict:
    """DSR (đề cương mục 11.6). sr_hat và sr_trials ở đơn vị theo ngày (không niên hóa); kurt là độ nhọn thông thường."""
    sr_trials = np.asarray(sr_trials, float)
    N = len(sr_trials)
    if N > 1:
        sd = float(np.std(sr_trials, ddof=1))
        sr_star = sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / N)
                        + EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (N * math.e)))
    else:
        sr_star = 0.0
    den = math.sqrt(max(1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr_star) * math.sqrt(max(T - 1, 1)) / den
    return {"sr_star": sr_star, "z": z, "dsr": float(stats.norm.cdf(z)), "N": N}


# ---------------------------------------------------------------- đánh giá phương sai/VaR
def kupiec(violations, alpha: float) -> dict:
    v = np.asarray(violations, bool)
    n, x = len(v), int(v.sum())
    if n == 0:
        return {"lr": float("nan"), "p": float("nan"), "x": 0, "n": 0}
    pi = x / n

    def ll(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return (n - x) * math.log(1 - p) + x * math.log(p)

    lr = -2 * (ll(alpha) - ll(pi))
    return {"lr": lr, "p": float(stats.chi2.sf(lr, 1)), "x": x, "n": n, "rate": pi}


def christoffersen_independence(violations) -> dict:
    v = np.asarray(violations, int)
    a, b = v[:-1], v[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())

    def xlogy(x, y):
        return 0.0 if x == 0 else x * math.log(max(y, 1e-12))

    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    lr = -2 * (ll0 - ll1)
    return {"lr": lr, "p": float(stats.chi2.sf(lr, 1)), "n01": n01, "n11": n11}


def ljung_box(x, lags: int = 10) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    rho = np.array([float((xc[k:] * xc[:-k]).sum()) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * float(np.sum(rho ** 2 / (n - np.arange(1, lags + 1))))
    return {"q": q, "p": float(stats.chi2.sf(q, lags)), "lags": lags, "n": n}


def newey_west_lrv(d, lag: int) -> float:
    d = np.asarray(d, float)
    d = d - d.mean()
    n = len(d)
    lrv = float((d ** 2).sum() / n)
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2 * (1 - k / (lag + 1)) * float((d[k:] * d[:-k]).sum() / n)
    return lrv


def dm_test(loss_a, loss_b, lag: int = 5) -> dict:
    """Diebold-Mariano: p một phía nhỏ nghĩa là a có tổn thất nhỏ hơn b (a tốt hơn)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    lrv = newey_west_lrv(d, lag)
    stat = float(d.mean() / math.sqrt(lrv / n)) if lrv > 0 else float("nan")
    return {"dm": stat, "p_a_better": float(stats.t.cdf(stat, df=n - 1)), "mean_diff": float(d.mean()), "n": n}
