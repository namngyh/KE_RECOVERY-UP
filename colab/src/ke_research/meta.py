"""S8 - M4: Meta-Labeling (đề cương mục 8.4).

Sự kiện e = lệnh mở mới của KE gốc (M0). Nhãn y_e = 1(net P&L của lệnh > 0) sau chi phí thật và cơ chế thoát của KE.
Z_e gồm các đặc trưng BIẾT TẠI nến tín hiệu: |B|, khoảng cách tới EMA, I~, A~, dI~, J~ = d*I~, cờ lịch, phần phiên, log h~.
Mô hình: hồi quy logistic L2, huấn luyện lại mỗi tháng, chỉ dùng các sự kiện đã KẾT THÚC trước đầu tháng trừ embargo
(purging theo thời điểm thoát lệnh: nhãn chồng lấn không được lọt vào tập huấn luyện).
Áp dụng: xác suất p được tính cho MỌI nến (cả hai phía) để làm bộ lọc lệnh mở mới; mô hình học trên các lệnh M0 thật nên áp
dụng cho nến ứng viên khác M0 là xấp xỉ (đã khai báo). Brier, AUC và lợi nhuận theo nhóm xác suất chỉ báo cáo cho dev/val.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats as sps

from strategies import ke_recovery as ke

from .models import DEFAULTS, Lab
from .prereg import PREREG, period_of

FEATURES = list(PREREG["M4"]["features"])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> dict:
    """Logistic L2: min mean(log-loss) + ||w||^2 / (2 C n), chuẩn hóa đặc trưng theo trung bình/độ lệch của tập huấn luyện."""
    mu = X.mean(axis=0); sd = X.std(axis=0); sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    n, k = Z.shape

    def obj(theta):
        w, b = theta[:k], theta[k]
        s = Z @ w + b
        p = _sigmoid(s)
        loss = np.mean(np.logaddexp(0, s) - y * s) + 0.5 / (C * n) * float(w @ w)
        g_s = (p - y) / n
        return loss, np.concatenate([Z.T @ g_s + w / (C * n), [g_s.sum()]])

    res = optimize.minimize(obj, np.zeros(k + 1), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return {"mu": mu, "sd": sd, "w": res.x[:k], "b": float(res.x[k])}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    return _sigmoid(((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = sps.rankdata(p)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def train_events(exit_times: pd.Series, month_start: pd.Timestamp, embargo_days: int) -> np.ndarray:
    """Mặt nạ các sự kiện được phép huấn luyện cho tháng bắt đầu ở month_start: đã thoát NGHIÊM NGẶT trước (month_start - embargo)."""
    return (exit_times < month_start - pd.Timedelta(days=embargo_days)).to_numpy()


def side_features(lab: Lab, side: int) -> np.ndarray:
    """Ma trận đặc trưng (n x len(FEATURES)) giả định mọi nến là tín hiệu vào lệnh phía `side` (+1 mua, -1 bán khống)."""
    p = DEFAULTS
    g = lab.g0("raw")
    f = ke._linreg(g, p["reg_period"])
    s = ke._wma(f, p["ma_period"])
    B = (f - s)
    absB = (B.abs() / lab.dev_std(B)).to_numpy()
    close = lab.raw["close"]
    ema_prev = ke._ema(close, p["ema_length"]).shift(1)
    sd_ret = np.log(close).diff().rolling(60, min_periods=20).std()
    dist = (side * (close - ema_prev) / (close * sd_ret)).to_numpy()
    F, bc = lab.F, lab.bc
    slot = lab.slot.to_numpy()
    part = np.where(slot < 11 * 60 + 31, 0.0, np.where(slot < 14 * 60 + 45, 1.0, 2.0))
    cols = {
        "absB_norm": absB, "dist_ema_z": dist, "I_tilde": F["I_tf"].to_numpy(), "A_tilde": F["A_tf"].to_numpy(),
        "dI_tilde": F["dI_in_tf"].to_numpy(), "J_tilde": side * F["I_tf"].to_numpy(),
        "near_expiry": bc["near_expiry"].to_numpy(float), "after_holiday": bc["after_holiday"].to_numpy(float),
        "first_day_after_expiry": bc["first_day_after_expiry"].to_numpy(float), "session_part": part,
        "logh_z": np.nan_to_num(lab.logh_z_bar(), nan=0.0), "side": np.full(lab.n, float(side)),
    }
    X = np.column_stack([cols[k] for k in FEATURES])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def meta_probabilities(lab: Lab, trades: pd.DataFrame, use_logh: bool = True) -> dict:
    """p_long, p_short theo nến (NaN khi chưa đủ sự kiện lịch sử -> không lọc) cùng chẩn đoán trên dev/val."""
    cfg = PREREG["M4"]
    X_long, X_short = side_features(lab, +1), side_features(lab, -1)
    if not use_logh:
        j = FEATURES.index("logh_z")
        X_long[:, j] = 0.0; X_short[:, j] = 0.0
    sig_bar = trades["entry_index"].to_numpy() - 1           # nến tín hiệu (khớp ở open nến kế tiếp)
    side = np.where(trades["side"].to_numpy() == "long", 1, -1)
    y = (trades["net_pnl"].to_numpy() > 0).astype(float)
    Xe = np.where((side == 1)[:, None], X_long[sig_bar], X_short[sig_bar])
    exit_t = trades["exit_time"]

    ts = lab.raw["ts"]
    month = ts.dt.to_period("M").to_numpy()
    p_long = np.full(lab.n, np.nan); p_short = np.full(lab.n, np.nan)
    p_event = np.full(len(trades), np.nan)
    ev_month = month[sig_bar]
    used = {}
    for m in pd.unique(month):
        idx = np.flatnonzero(month == m)
        start = ts.iloc[idx[0]].normalize().replace(day=1)
        mask = train_events(exit_t, start, cfg["embargo_sessions"])
        if mask.sum() < cfg["min_events"]:
            continue
        model = fit_logistic(Xe[mask], y[mask], cfg["C"])
        p_long[idx] = predict(model, X_long[idx]); p_short[idx] = predict(model, X_short[idx])
        used[str(m)] = int(mask.sum())
        sel = np.flatnonzero(ev_month == m)
        if len(sel):
            p_event[sel] = predict(model, Xe[sel])
    diag = {"months_with_model": len(used), "events_total": int(len(trades)),
            "first_month_with_model": min(used) if used else None}
    per = period_of(trades["entry_time"]).to_numpy()
    for pname in ("dev", "val"):
        m = (per == pname) & np.isfinite(p_event)
        if m.sum() > 20:
            pe, ye = p_event[m], y[m]
            groups = pd.qcut(pe, 3, labels=False, duplicates="drop")
            net = trades["net_points"].to_numpy()[m]
            diag[pname] = {"n": int(m.sum()), "brier": float(np.mean((pe - ye) ** 2)),
                           "brier_base": float(np.mean((ye.mean() - ye) ** 2)), "auc": auc(ye, pe),
                           "mean_net_points_by_prob_tercile": [float(net[groups == k].mean()) for k in sorted(set(groups))],
                           "win_rate_by_prob_tercile": [float(ye[groups == k].mean()) for k in sorted(set(groups))]}
    return {"p_long": p_long, "p_short": p_short, "diag": diag}
