"""Các bước S3-S7 của pipeline (đề cương mục 5-10). Mỗi hàm nhận Context (xem pipeline.Context) và trả về tóm tắt ngắn.

Kỷ luật tập test: mọi bước ở đây chỉ báo cáo kết quả trên dev và val. Dữ liệu test chỉ được dùng ở S9 (steps_final.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import calendar_feats as cf
from . import egarchx, features, models, predictive, prereg, stats
from .experiments import Experiment
from .prereg import PREREG, period_of

KE_PARAMS = models.DEFAULTS
COLS = ["ts", "open_time_ms", "open", "high", "low", "close"]


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


def load_lab(ctx, with_egarch: bool = True) -> models.Lab:
    """Dựng Lab từ kết quả S0-S3 (và S7 nếu có): giá thực, giá hiệu chỉnh, đặc trưng, lịch theo nến, dự báo EGARCH-X."""
    raw0 = ctx.load("S0", "bars_5m")
    adj = ctx.load("S2", "bars_5m_adj")
    F = ctx.load("S3", "features_5m")
    cal = ctx.load("S2", "session_calendar").set_index("day")
    bc = cf.add_bar_calendar(raw0, cal)
    eg = None
    if with_egarch and ctx.exists("S7", "egarch_daily"):
        eg = ctx.load("S7", "egarch_daily").set_index("day")
    return models.Lab(raw0[COLS], adj[COLS], F, bc, eg)


# ------------------------------------------------------------------------ S3
def step_s3(ctx) -> dict:
    """Đặc trưng mua/bán (mục 6): I, A, dI (hai phiên bản), chuẩn hóa theo ô, ER."""
    adj = ctx.load("S2", "bars_5m_adj")
    F = features.build_features(adj)
    # cổng 1: không rò rỉ tương lai - đổi toàn bộ khối lượng và giá sau nến `cut` thì đặc trưng trước đó không đổi
    cut = len(adj) // 2
    adj2 = adj.copy()
    vcols = ["volume", "buy_vol", "sell_vol"]
    adj2.loc[cut + 1:, vcols] = adj2.loc[cut + 1:, vcols] * 3 + 50
    adj2.loc[cut + 1:, "close"] = adj2.loc[cut + 1:, "close"] * 1.7
    F2 = features.build_features(adj2)
    for c in ("I", "A", "dI_in", "dI_ov", "I_t", "A_t", "dI_in_t", "dI_ov_t", "ER"):
        if not np.allclose(F[c].iloc[:cut + 1].fillna(-9e9), F2[c].iloc[:cut + 1].fillna(-9e9)):
            raise AssertionError(f"Rò rỉ tương lai ở đặc trưng {c}")
    # cổng 2: miền giá trị
    if not F["I"].dropna().between(-1, 1).all():
        raise AssertionError("I ngoài [-1, 1]")
    atc = F["slot"] == features.ATC_SLOT
    rep = {
        "bars": int(len(F)), "unclassified_bars": int(F["unclassified"].sum()),
        "I_nan_share": float(F["I"].isna().mean()), "dI_in_nan_share": float(F["dI_in"].isna().mean()),
        "dI_ov_nan_share": float(F["dI_ov"].isna().mean()),
        "atc_bars": int(atc.sum()),
        "I_mean_atc": float(F.loc[atc, "I"].mean()), "I_std_atc": float(F.loc[atc, "I"].std()),
        "I_mean_non_atc": float(F.loc[~atc, "I"].mean()), "I_std_non_atc": float(F.loc[~atc, "I"].std()),
        "I_tilde_std_atc": float(F.loc[atc, "I_t"].std()), "I_tilde_std_non_atc": float(F.loc[~atc, "I_t"].std()),
        "A_tilde_std_atc": float(F.loc[atc, "A_t"].std()), "A_tilde_std_non_atc": float(F.loc[~atc, "A_t"].std()),
        "max_abs_standardized": {c: float(F[c].abs().max()) for c in ("I_t", "A_t", "dI_in_t", "dI_ov_t")},
        "causality_check": "đã đổi toàn bộ khối lượng và giá sau nến giữa mẫu: đặc trưng trước đó không đổi",
        "missing_rule": PREREG["features"]["missing_rule"],
    }
    ctx.save(F, "S3", "features_5m")
    write_json(os.path.join(ctx.out("S3"), "report.json"), rep)
    return {k: rep[k] for k in ("bars", "unclassified_bars", "I_std_atc", "I_std_non_atc",
                                "I_tilde_std_atc", "I_tilde_std_non_atc")}


# ------------------------------------------------------------------------ S4
def step_s4(ctx) -> dict:
    """Đăng ký trước: ghi preregistration.json (kèm SHA-256) và mô tả chia mẫu."""
    lab = load_lab(ctx, with_egarch=False)
    prereg.write(ctx.out("S4"))
    days = pd.Series(lab.day.unique())
    per = period_of(days)
    split = {p: {"days": int((per == p).sum()), "from": str(days[per == p].min().date()),
                 "to": str(days[per == p].max().date())} for p in ("dev", "val", "test")}
    write_json(os.path.join(ctx.out("S4"), "split.json"), split)
    return {"prereg_sha256": prereg.digest()[:16],
            "split": {p: f"{v['from']}..{v['to']} ({v['days']} phiên)" for p, v in split.items()}}


# ------------------------------------------------------------------------ S5
def step_s5(ctx) -> dict:
    """Kiểm định H1-H4, hồi quy gia tăng, IC (mục 10); chỉ dev và val."""
    prereg.verify(ctx.root)
    lab = load_lab(ctx, with_egarch=False)
    q = models.build_signal(lab, {})
    out = predictive.run_all(lab, q)
    write_json(os.path.join(ctx.out("S5"), "predictive.json"), out)
    h6 = out["h6"]
    key = {f"{r['test']}_{r['period']}": round(r["p"], 4) for r in h6["groups"] + h6["calendar"]
           if r["test"] != "H3_signed_extra"}
    inc = {r["period"]: round(r["p"], 4) for r in h6["increment"]}
    return {"primary_h": 6, "n_obs_h6": h6["n_obs"], "p_values_h6": key, "increment_wald_p": inc}


# ------------------------------------------------------------------------ S6
def ablation_specs() -> dict:
    """A0-A7 (mục 10.4). Phép 'bỏ trực tiếp' và, khi có ý nghĩa, phương án tái hiệu chỉnh chỉ trên dev."""
    specs = {"A0_KE_goc": {}, "A1_RSI_only": {"g_mode": "rsi_only"}, "A2_Stoch_only": {"g_mode": "k_only"},
             "A3_bo_EMA": {"ema_gate": False}, "A6_long_only": {"side": "long"}, "A6_short_only": {"side": "short"},
             "A7_dong_cuoi_phien": {"flat_eod": True}}
    for sm in ("wma", "sma"):                      # A4: thay LinReg; tái hiệu chỉnh độ dài cửa sổ trên dev
        for L in (32, 48, 64, 96):
            specs[f"A4_{sm}_L{L}"] = {"smoother": sm, "params": {"reg_period": L}}
    for name, base in KE_PARAMS.items():           # A5: tham số lân cận +-10%
        for f in (0.9, 1.1):
            v = base * f
            v = round(v, 2) if name == "n1" else int(round(v))
            specs[f"A5_{name}_{v}"] = {"params": {name: v}}
    return specs


def step_s6(ctx) -> dict:
    prereg.verify(ctx.root)
    lab = load_lab(ctx, with_egarch=False)
    ex = Experiment(lab)
    rows, runs = [], {}
    for name, spec in ablation_specs().items():
        r = ex.run(spec)
        runs[name] = r
        m = ex.metrics(r)
        rows.append({"config": name, **{f"{p}_{k}": v for p, d in m.items() for k, v in d.items()}})
    tab = pd.DataFrame(rows)
    ctx.save(tab, "S6", "ablation_table")
    mat = ex.matrix(runs)
    mat.index.name = "day"
    ctx.save(mat.reset_index(), "S6", "daily_dev_val")
    a0 = tab[tab["config"] == "A0_KE_goc"].iloc[0]
    expo = {n: float((r["signal"] != 0).mean()) for n, r in runs.items()
            if n in ("A0_KE_goc", "A6_long_only", "A6_short_only")}
    summ = {"configs": len(tab), "A0_U_dev": round(float(a0["dev_U"]), 3), "A0_U_val": round(float(a0["val_U"]), 3),
            "exposure_share": {k: round(v, 3) for k, v in expo.items()},
            "best_dev_config": tab.sort_values("dev_U", ascending=False).iloc[0]["config"]}
    write_json(os.path.join(ctx.out("S6"), "report.json"), summ)
    return summ


# ------------------------------------------------------------------------ S7
def step_s7(ctx) -> dict:
    """EGARCH-X: ước lượng, dự báo ngoài mẫu, đánh giá trên dev/val, cổng so với đối chứng, H5."""
    prereg.verify(ctx.root)
    cfg = PREREG["egarchx"]
    bars1m = ctx.load("S0", "bars_1m")
    ret = ctx.load("S2", "session_returns").set_index("day")
    cal = ctx.load("S2", "session_calendar").set_index("day")
    d = egarchx.daily_inputs(bars1m, ret, cal)
    X = egarchx.design_matrix(d)
    r = d["r"].to_numpy(float)
    rv = d["rv_proxy"].to_numpy(float)
    per = period_of(pd.Series(d.index, index=d.index)).to_numpy()
    fcs = {k: egarchx.rolling_forecasts(k, r, X, cfg["min_train_sessions"], cfg["refit_every"])
           for k in cfg["distributions"]}
    bm = egarchx.benchmark_variances(r, rv)

    evals = {}
    for p in ("dev", "val"):
        base = (per == p)
        evals[p] = {k: egarchx.evaluate(fc, r, rv, base) for k, fc in fcs.items()}
        mask = base & np.isfinite(fcs["normal"]["h"].to_numpy())
        bench = {}
        for n in bm:
            hh = bm[n].to_numpy()
            ok = mask & np.isfinite(hh)
            bench[n] = {"qlike": float(np.mean(egarchx.qlike(hh[ok], rv[ok])))}
        evals[p]["benchmarks"] = bench
    # chọn phân phối theo log-score nhỏ nhất trên val (prereg)
    chosen = min(cfg["distributions"], key=lambda k: evals["val"][k]["log_score"])
    # cổng: QLIKE của mô hình đã chọn nhỏ hơn có ý nghĩa so với đối chứng tốt nhất trên val (DM một phía p<0,10)
    vm = (per == "val") & np.isfinite(fcs[chosen]["h"].to_numpy())
    best_bm = min(bm.columns, key=lambda n: evals["val"]["benchmarks"][n]["qlike"])
    ok = vm & np.isfinite(bm[best_bm].to_numpy())
    dm = stats.dm_test(egarchx.qlike(fcs[chosen]["h"].to_numpy()[ok], rv[ok]),
                       egarchx.qlike(bm[best_bm].to_numpy()[ok], rv[ok]))
    gate = bool(dm["p_a_better"] < 0.10)

    table = pd.DataFrame({"day": d.index, "r": r, "rv_proxy": rv, "h_forecast": fcs[chosen]["h"].to_numpy(),
                          "z_sel": fcs[chosen]["z"].to_numpy(), "u_sel": fcs[chosen]["u"].to_numpy()})
    for k, fc in fcs.items():
        table[f"h_{k}"] = fc["h"].to_numpy()
        table[f"ls_{k}"] = fc["ls"].to_numpy()
    for n in bm:
        table[f"bm_{n}"] = bm[n].to_numpy()
    ctx.save(table, "S7", "egarch_daily")

    lab = load_lab(ctx, with_egarch=True)
    q = models.build_signal(lab, {})
    h5 = predictive.h5_test(predictive.build_panel(lab, q, PREREG["horizons"]["primary"]))
    rep = {"chosen_distribution": chosen, "evaluation": evals, "best_benchmark_val": best_bm,
           "dm_vs_best_benchmark_val": dm, "gate_passed": gate, "gate_rule": cfg["gate"], "H5": h5,
           "fits_converged": {k: float(np.mean([f["converged"] for f in fc.attrs["fits"]])) for k, fc in fcs.items()},
           "consequence": "M1-M4 đủ điều kiện vào M9" if gate
           else "KHÔNG qua cổng: M1-M4 chạy như kiểm chứng âm, không vào M9"}
    write_json(os.path.join(ctx.out("S7"), "report.json"), rep)
    return {"chosen": chosen, "best_benchmark_val": best_bm,
            "qlike_val_egarch": round(evals["val"][chosen]["qlike"], 4),
            "qlike_val_benchmark": round(evals["val"]["benchmarks"][best_bm]["qlike"], 4),
            "dm_p": round(dm["p_a_better"], 4), "gate_passed": gate}
