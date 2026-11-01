"""S5 - kiểm định giả thuyết H1-H5 và thông tin gia tăng (đề cương mục 3.4, 10.1-10.3).

Y_{t,h} = q_t^KE * R_{t,h}^exec : lợi suất có dấu theo hướng KE, vào ở open nến t+1, ra ở open nến t+1+h CÙNG phiên,
đã trừ trượt giá 2 x 0,2 điểm và phí 2 x 0,005% (lợi suất 'khả thi', mục 3.4).
Mọi suy luận dùng sai số chuẩn theo cụm PHIÊN (mục 10.1: HAC/cụm vì các quan sát chồng lấn). Ngưỡng tercile lấy từ dev.
H1: E[Y|J cao] > E[Y|J thấp] (một phía) | H2: A cao != A thấp | H3: dI cao != dI thấp | H4: phụ thuộc lịch | H5: phụ thuộc h^.
Chỉ dev và val được dùng; tập test không bao giờ vào module này.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats as st
from .models import Lab
from .prereg import PREREG

FEE = PREREG["costs"]["fee_rate"]
SLIP = PREREG["costs"]["slippage_points"]


def eligible_returns(lab: Lab, h: int):
    """R_exec (chưa nhân q), chi phí vòng khứ hồi theo tỷ lệ, và mặt nạ nến đủ điều kiện (vào/ra cùng phiên với tín hiệu)."""
    n = lab.n
    o = lab.raw["open"].to_numpy(float)
    day = lab.day.to_numpy()
    idx = np.arange(n)
    t1 = np.minimum(idx + 1, n - 1)
    t2 = np.minimum(idx + 1 + h, n - 1)
    ok = (idx + 1 + h < n) & (day[idx] == day[t2])
    return (o[t2] - o[t1]) / o[t1], 2 * SLIP / o[t1] + 2 * FEE, ok


def build_panel(lab: Lab, q: np.ndarray, h: int, periods=("dev", "val")) -> pd.DataFrame:
    """Bảng quan sát (mọi nến đủ điều kiện thuộc `periods`) với biến giải thích và điều khiển (mục 10.3)."""
    R, cost, ok = eligible_returns(lab, h)
    F = lab.F
    close = lab.adj["close"]
    lr = np.log(close).diff()
    slot = lab.slot.to_numpy()
    part = np.where(slot < 11 * 60 + 31, 0, np.where(slot < 14 * 60 + 45, 1, 2))     # sáng / chiều / ATC
    keep = ok & np.isin(lab.period, periods)
    bc = lab.bc
    df = pd.DataFrame({
        "day": lab.day.to_numpy(), "period": lab.period, "q": q, "R": R, "Y": q * R - cost,
        "I": F["I"].to_numpy(), "A": F["A"].to_numpy(), "dI_in": F["dI_in"].to_numpy(), "dI_ov": F["dI_ov"].to_numpy(),
        "I_tf": F["I_tf"].to_numpy(), "A_tf": F["A_tf"].to_numpy(),
        "dI_in_tf": F["dI_in_tf"].to_numpy(), "dI_ov_tf": F["dI_ov_tf"].to_numpy(),
        "unclassified": F["unclassified"].to_numpy(float),
        "near_expiry": bc["near_expiry"].to_numpy(float), "after_holiday": bc["after_holiday"].to_numpy(float),
        "first_day": bc["first_day_after_expiry"].to_numpy(float), "after_weekend": bc["after_weekend"].to_numpy(float),
        "part": part,
        "mom": (np.log(close) - np.log(close.shift(6))).to_numpy() * 100,
        "vol": lr.rolling(60, min_periods=20).std().to_numpy() * 100,
        "logh_z": lab.logh_z_bar(),
    })
    df["J"] = df["q"] * df["I"]
    df["qdI_in"] = df["q"] * df["dI_in"]
    return df[keep].reset_index(drop=True)


def _thresholds(panel: pd.DataFrame, col: str, tercile: float):
    x = panel.loc[(panel["period"] == "dev") & (panel["q"] != 0), col].dropna()
    return float(x.quantile(tercile)), float(x.quantile(1 - tercile))


def group_tests(panel: pd.DataFrame) -> list[dict]:
    """H1-H3 (và biến thể có dấu của dI) cho từng giai đoạn, ngưỡng tercile từ dev."""
    tcl = PREREG["tests"]["tercile"]
    rows = []
    specs = [("H1", "J", True, "J=q*I, một phía (cao > thấp)"), ("H2", "A", False, "A hai phía"),
             ("H3_in", "dI_in", False, "dI trong cùng phiên, hai phía"),
             ("H3_ov", "dI_ov", False, "dI cho phép qua đêm, hai phía"),
             ("H3_signed_extra", "qdI_in", False, "BỔ SUNG (ngoài đề cương): q*dI trong phiên, hai phía")]
    for name, col, one_sided, note in specs:
        lo_thr, hi_thr = _thresholds(panel, col, tcl)
        for period in ("dev", "val"):
            sub = panel[(panel["period"] == period) & (panel["q"] != 0)]
            if sub.empty:
                continue
            r = st.mean_diff(sub["Y"], sub[col] >= hi_thr, sub[col] <= lo_thr, sub["day"], one_sided=one_sided)
            rows.append({"test": name, "period": period, "var": col, "note": note,
                         "lo_thr": lo_thr, "hi_thr": hi_thr, **r})
    return rows


def calendar_test(panel: pd.DataFrame) -> list[dict]:
    """H4: Y phụ thuộc nhóm lịch; Wald chung của các biến giả (mục 10.3)."""
    rows = []
    cols = ["near_expiry", "after_holiday", "first_day", "after_weekend"]
    for period in ("dev", "val"):
        sub = panel[(panel["period"] == period) & (panel["q"] != 0)]
        if sub.empty:
            continue
        X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy() for c in cols])
        fit = st.cluster_ols(sub["Y"], X, sub["day"])
        w = st.wald_test(fit["beta"], fit["cov"], range(1, X.shape[1]))
        rows.append({"test": "H4", "period": period, "wald": w["stat"], "df": w["df"], "p": w["p"],
                     "betas": {c: float(fit["beta"][i + 1]) for i, c in enumerate(cols)},
                     "n": int(len(sub)), "clusters": int(fit["clusters"])})
    return rows


def h5_test(panel: pd.DataFrame, bins: int = 5) -> list[dict]:
    """H5: E[Y | h^] không hằng: nhóm theo phân vị của log h^ chuẩn hóa (ngưỡng từ dev), Wald chung các biến giả nhóm."""
    rows = []
    dev = panel[(panel["period"] == "dev") & (panel["q"] != 0)]["logh_z"].dropna()
    if len(dev) < 500:
        return rows
    edges = np.quantile(dev, np.linspace(0, 1, bins + 1)[1:-1])
    for period in ("dev", "val"):
        sub = panel[(panel["period"] == period) & (panel["q"] != 0) & panel["logh_z"].notna()]
        if len(sub) < 200:
            continue
        b = np.digitize(sub["logh_z"].to_numpy(), edges)
        X = np.column_stack([np.ones(len(sub))] + [(b == k).astype(float) for k in range(1, bins)])
        fit = st.cluster_ols(sub["Y"], X, sub["day"])
        w = st.wald_test(fit["beta"], fit["cov"], range(1, bins))
        rows.append({"test": "H5", "period": period, "wald": w["stat"], "df": w["df"], "p": w["p"], "n": int(len(sub)),
                     "mean_Y_by_bin": [float(sub["Y"].to_numpy()[b == k].mean()) for k in range(bins)]})
    return rows


def increment_regression(panel: pd.DataFrame) -> list[dict]:
    """Phương trình (increment): Y = a + b_I J~ + b_A A~ + b_D dI~ + g'C, so với mô hình chỉ có C; Wald chung b_I=b_A=b_D=0."""
    rows = []
    ctrl = ["q", "mom", "vol", "near_expiry", "after_holiday", "first_day", "unclassified"]
    for period in ("dev", "val"):
        sub = panel[(panel["period"] == period) & (panel["q"] != 0)].dropna(subset=ctrl)
        if len(sub) < 500:
            continue
        sub = sub.assign(J_tf=sub["q"] * sub["I_tf"], afternoon=(sub["part"] == 1).astype(float),
                         atc=(sub["part"] == 2).astype(float))
        C = ctrl + ["afternoon", "atc"]
        Xr = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy() for c in C])
        Xf = np.column_stack([Xr, sub["J_tf"].to_numpy(), sub["A_tf"].to_numpy(), sub["dI_in_tf"].to_numpy()])
        fr = st.cluster_ols(sub["Y"], Xr, sub["day"])
        ff = st.cluster_ols(sub["Y"], Xf, sub["day"])
        k = Xr.shape[1]
        w = st.wald_test(ff["beta"], ff["cov"], [k, k + 1, k + 2])
        rows.append({"test": "increment", "period": period, "n": int(len(sub)), "wald": w["stat"], "p": w["p"],
                     "r2_restricted": fr["r2"], "r2_full": ff["r2"], "delta_r2": ff["r2"] - fr["r2"],
                     "beta": {"J": float(ff["beta"][k]), "A": float(ff["beta"][k + 1]), "dI": float(ff["beta"][k + 2])},
                     "p_coef": {"J": float(ff["p"][k]), "A": float(ff["p"][k + 1]), "dI": float(ff["p"][k + 2])}})
    return rows


def daily_ic(panel: pd.DataFrame, min_obs: int = 10) -> list[dict]:
    """IC = Corr(S_t, R^exec) kiểu Spearman, tính theo từng phiên rồi lấy trung bình và t giữa các phiên (mục 10.5)."""
    rows = []
    for period in ("dev", "val"):
        sub = panel[panel["period"] == period]
        for var, tgt in (("I", "R"), ("dI_in", "R"), ("J", "Y")):
            ics = []
            for _, g in sub.groupby("day"):
                g = g.dropna(subset=[var, tgt])
                if len(g) >= min_obs and g[var].nunique() > 1 and g[tgt].nunique() > 1:
                    ics.append(sps.spearmanr(g[var], g[tgt]).statistic)
            ics = np.array(ics)
            if len(ics) > 20:
                rows.append({"test": "IC", "period": period, "var": var, "target": tgt, "mean_ic": float(ics.mean()),
                             "t_across_days": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))),
                             "days": int(len(ics))})
    return rows


def run_all(lab: Lab, q: np.ndarray) -> dict:
    """Chạy H1-H4, hồi quy gia tăng, IC cho mọi chân trời; Holm trong từng chân trời cho họ kiểm định chính."""
    out = {}
    for h in PREREG["horizons"]["bars"]:
        panel = build_panel(lab, q, h)
        g = group_tests(panel)
        cal = calendar_test(panel)
        inc = increment_regression(panel)
        for period in ("dev", "val"):
            fam = [r for r in g + cal if r["period"] == period and r["test"] != "H3_signed_extra"]
            adj = st.holm([r["p"] for r in fam]) if fam else []
            for r, a in zip(fam, adj):
                r["p_holm"] = float(a)
        out[f"h{h}"] = {"groups": g, "calendar": cal, "increment": inc, "ic": daily_ic(panel),
                        "n_obs": int(((panel["period"].isin(["dev", "val"])) & (panel["q"] != 0)).sum())}
    return out
