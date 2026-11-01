"""S8-S9: xây M1-M9 trên cùng engine và đánh giá cuối (đề cương mục 8, 11).

S8 chỉ báo cáo dev và val; chọn cấu hình tốt nhất TRONG từng họ bằng U trên dev (prereg).
S9 áp luật qua cổng đã đăng ký trên val, dựng M9, tính SPA/PBO/DSR, rồi chạy tập test ĐÚNG MỘT LẦN cho tập mô hình đã khóa
và ghi TEST_LOCK.json. Chạy lại test cần ctx.allow_test_rerun = True và sẽ được ghi nhận.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import features, meta, prereg, runner, stats
from .experiments import Experiment
from .models import Lab
from .prereg import PREREG
from .steps import load_lab, write_json

FAMILY_ORDER = ["M1", "M2", "M3", "M4", "M5A", "M5B", "M5C", "M6", "M7", "M8"]
EGARCH_FAMILIES = {"M1", "M2", "M3", "M4"}


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def egarch_gate(ctx) -> bool:
    path = os.path.join(ctx.root, "results", "S7", "report.json")
    return bool(read_json(path)["gate_passed"]) if os.path.exists(path) else False


def build_families(sd: float) -> dict:
    P = PREREG
    cs = P["lambda_grid_c"]
    fam = {
        "M1": [(f"M1_c{c:+g}", {"add": [["logh_z", c * sd]]}) for c in [-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]],
        "M2": [("M2_band", {"gates": [{"t": "vol_band", "lo": P["filters"]["M2_band_q"][0],
                                       "hi": P["filters"]["M2_band_q"][1]}]})],
        "M3": [("M3_upper", {"gates": [{"t": "vol_upper", "hi": P["filters"]["M3_upper_q"]}]})],
        "M4": [(f"M4_tau{t}", {"gates": [{"t": "meta", "tau": t}]}) for t in P["M4"]["tau_grid"]],
        "M5A": [("M5A_adj", {"price": "adj"})],
        "M5B": [("M5B_season", {"g_mode": "m5b"})],
        "M5C": [(f"M5C_{v}", {"gates": [{"t": "cal", "variant": v}]}) for v in P["filters"]["M5C_variants"]],
        "M7": [(f"M7_q{q}", {"gates": [{"t": "er_min", "q": q}]}) for q in P["filters"]["M7_er_q"]],
        "M8": [("M8_adaptive", {"adaptive": True})],
    }
    g6 = []
    for c in cs:
        lam = c * sd
        r2 = lam / np.sqrt(2)
        g6 += [(f"M6_G1_c{c}", {"add": [["I_tf", lam]]}), (f"M6_G2_c{c}", {"add": [["A_tf", lam]]}),
               (f"M6_G3in_c{c}", {"add": [["dI_in_tf", lam]]}), (f"M6_G3ov_c{c}", {"add": [["dI_ov_tf", lam]]}),
               (f"M6_G4_c{c}", {"add": [["I_tf", r2], ["dI_in_tf", r2]]})]
    fam["M6"] = g6
    return fam


def control_specs(sd: float) -> list:
    """Đối chứng/độ nhạy: KHÔNG đủ điều kiện vào bước chọn họ hay M9, chỉ để diễn giải."""
    return [("M8_ctrl_fixed_32_20", {"params": {"reg_period": 32, "ma_period": 20}}),
            ("M8_ctrl_fixed_96_40", {"params": {"reg_period": 96, "ma_period": 40}}),
            ("G5_aux_season_plus_I", {"g_mode": "m5b", "add": [["I_tf", 0.25 * sd]]})]


def prepare(ctx):
    """Lab + Experiment + chạy M0 + xác suất meta-label (đặt vào lab.meta_p)."""
    lab = load_lab(ctx, with_egarch=True)
    ex = Experiment(lab)
    m0 = ex.run({})
    mp = meta.meta_probabilities(lab, m0["trades"], use_logh=egarch_gate(ctx))
    lab.meta_p = (mp["p_long"], mp["p_short"])
    return lab, ex, m0, mp["diag"]


def _swap_adj(lab: Lab, new_adj: pd.DataFrame):
    old = lab.adj
    lab.adj = new_adj.reset_index(drop=True)
    for k in [k for k in lab._cache if isinstance(k, tuple) and k[:2] == ("rsik", "adj")]:
        del lab._cache[k]
    return old


def step_s8(ctx) -> dict:
    prereg.verify(ctx.root)
    lab, ex, m0, meta_diag = prepare(ctx)
    gate = egarch_gate(ctx)
    sd = lab.g0_sd_dev()
    fams = build_families(sd)
    rows, runs = [], {"M0": m0}
    m = ex.metrics(m0)
    rows.append({"family": "M0", "config": "M0", "spec": "{}", "eligible": True,
                 **{f"{p}_{k}": v for p, d in m.items() for k, v in d.items()}})
    diag = {"M4_meta": meta_diag}

    def add(family, name, spec, eligible):
        r = ex.run(spec)
        runs[name] = r
        mm = ex.metrics(r)
        rows.append({"family": family, "config": name, "spec": json.dumps(spec, sort_keys=True),
                     "eligible": eligible, **{f"{p}_{k}": v for p, d in mm.items() for k, v in d.items()}})
        return r

    for fam in FAMILY_ORDER:
        for name, spec in fams[fam]:
            add(fam, name, spec, eligible=(fam not in EGARCH_FAMILIES) or gate)
            if fam == "M8":
                diag["M8_regime_artifacts"] = dict(lab.diag)
    for name, spec in control_specs(sd):
        add("control", name, spec, eligible=False)

    # độ nhạy: M5A với Delta ổn định (xử lý riêng lần roll bất thường)
    off = ctx.load("S2", "offset_robust")
    robust = lab.raw[["ts", "open_time_ms"]].copy()
    for c in ("open", "high", "low", "close"):
        robust[c] = lab.raw[c].to_numpy() - off["cum_offset_robust"].to_numpy()
    old = _swap_adj(lab, robust)
    add("sensitivity", "M5A_adj_robustDelta", {"price": "adj", "_tag": "robust"}, eligible=False)
    _swap_adj(lab, old)

    best = {}
    for fam in FAMILY_ORDER:
        cand = [rw for rw in rows if rw["family"] == fam and np.isfinite(rw["dev_U"])]
        if cand:
            b = max(cand, key=lambda rw: rw["dev_U"])
            best[fam] = {"config": b["config"], "spec": json.loads(b["spec"]), "eligible": bool(b["eligible"]),
                         "dev_U": b["dev_U"], "val_U": b["val_U"]}

    if "M6" in best:                                   # độ nhạy bổ sung (ngoài đề cương): bỏ ATC khỏi I, A, dI
        saved = lab.F
        lab.F = features.build_features(ctx.load("S2", "bars_5m_adj"), exclude_atc=True)
        mm = ex.metrics(ex.run(dict(best["M6"]["spec"], _tag="exATC")))
        diag["M6_best_excluding_ATC_sensitivity"] = {"config": best["M6"]["config"], "dev_U": mm["dev"]["U"],
                                                       "val_U": mm["val"]["U"],
                                                       "note": "độ nhạy bổ sung (không có trong đề cương)"}
        lab.F = saved

    # nhiễu bước theo ngày của M1/M2/M3: tỉ lệ thay đổi vị thế ở 2 nến đầu phiên
    first2 = lab.raw.groupby(lab.day).cumcount().to_numpy() < 2

    def first_share(sig):
        ch = np.concatenate([[False], sig[1:] != sig[:-1]])
        return float((ch & first2).sum() / max(ch.sum(), 1))

    diag["position_change_share_first2_bars"] = {
        "M0": first_share(m0["signal"]),
        **{f: first_share(runs[best[f]["config"]]["signal"]) for f in ("M1", "M2", "M3") if f in best}}
    tab = pd.DataFrame(rows)
    ctx.save(tab, "S8", "models_table")
    mat = ex.matrix(runs)
    mat.index.name = "day"
    ctx.save(mat.reset_index(), "S8", "daily_dev_val")
    write_json(os.path.join(ctx.out("S8"), "selection.json"),
               {"egarch_gate": gate, "best_per_family": best, "diagnostics": diag})
    return {"configs": len(tab), "egarch_gate": gate,
            "best_per_family": {f: f"{b['config']} (U dev {b['dev_U']:.2f} / val {b['val_U']:.2f})" for f, b in best.items()},
            "M0": f"U dev {rows[0]['dev_U']:.2f} / val {rows[0]['val_U']:.2f}"}


# ------------------------------------------------------------------------ S9
def compose(specs: list[dict]) -> dict:
    """M9: ghép các cải tiến đã qua cổng (giá hiệu chỉnh, chuẩn hóa lịch, cộng đặc trưng vào g, bộ lọc, thích nghi)."""
    out = {"add": [], "gates": []}
    for s in specs:
        if s.get("price") == "adj":
            out["price"] = "adj"
        if s.get("g_mode") == "m5b":
            out["g_mode"] = "m5b"
        if s.get("adaptive"):
            out["adaptive"] = True
        out["add"] += s.get("add", [])
        out["gates"] += s.get("gates", [])
    return {k: v for k, v in out.items() if v}


def compare(ex: Experiment, a: dict, base: dict, period: str, seed: int) -> dict:
    da, db = ex.daily_slice(a, period).to_numpy(), ex.daily_slice(base, period).to_numpy()
    gb = PREREG["gate"]["paired_bootstrap"]
    r = stats.paired_block_bootstrap(da, db, gb["block_sessions"], gb["reps"], seed)
    r["mdd_ratio"] = stats.max_drawdown(da) / max(stats.max_drawdown(db), 1e-9)
    return r


def daily_sr(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def dsr_for(daily: np.ndarray, sr_trials: np.ndarray) -> dict:
    sr = daily_sr(daily)
    return stats.deflated_sharpe(sr, sr_trials[np.isfinite(sr_trials)], len(daily), float(sps.skew(daily)),
                                 float(sps.kurtosis(daily, fisher=False)))


def md_table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            numeric = isinstance(v, (int, float, np.floating)) and not isinstance(v, (bool, np.bool_))
            cells.append(fmt.get(c, "{}").format(v) if numeric else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def step_s9(ctx) -> dict:
    prereg.verify(ctx.root)
    sel = read_json(os.path.join(ctx.root, "results", "S8", "selection.json"))
    gate = bool(sel["egarch_gate"])
    best = sel["best_per_family"]
    lab, ex, m0, _ = prepare(ctx)
    gb = PREREG["gate"]
    runs = {f: ex.run(b["spec"]) for f, b in best.items()}
    out = ctx.out("S9")

    # --- 1. luật qua cổng trên val (đã đăng ký trước)
    rows = []
    for i, f in enumerate(FAMILY_ORDER):
        if f not in best:
            continue
        c = compare(ex, runs[f], m0, "val", gb["paired_bootstrap"]["seed"] + i)
        rows.append({"family": f, "config": best[f]["config"], "eligible": bool(best[f]["eligible"]),
                     "dU_val": c["dU"], "p": c["p_one_sided"], "ci_lo": c["ci95"][0], "ci_hi": c["ci95"][1],
                     "mdd_ratio": c["mdd_ratio"], "U_val": c["U_a"], "U_M0_val": c["U_b"]})
    adj = stats.holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        r["passed"] = bool(r["eligible"] and r["dU_val"] > 0 and a < gb["holm_alpha"] and r["mdd_ratio"] <= gb["mdd_tolerance"])
    passed = [r["family"] for r in rows if r["passed"]]

    # --- 2. M9
    locked = {"M0": {}}
    for f in passed:
        locked[f] = best[f]["spec"]
    if len(passed) >= 2:
        spec9 = compose([best[f]["spec"] for f in passed])
        r9 = ex.run(spec9)
        c9 = compare(ex, r9, m0, "val", gb["paired_bootstrap"]["seed"] + 99)
        m9 = {"spec": spec9, "components": passed, "dU_val": c9["dU"], "p": c9["p_one_sided"],
              "mdd_ratio": c9["mdd_ratio"], "U_val": c9["U_a"], "U_dev": ex.metrics(r9)["dev"]["U"],
              "passed": bool(c9["dU"] > 0 and c9["p_one_sided"] < gb["holm_alpha"] and c9["mdd_ratio"] <= gb["mdd_tolerance"])}
        if m9["passed"]:
            locked["M9"] = spec9
    else:
        m9 = {"note": f"M9 không dựng: chỉ {len(passed)} cải tiến qua cổng (cần >= 2 để ghép)"}

    # --- 3. SPA trên val
    val_mask = (ex.day_period == "val").to_numpy()
    val_idx = ex.day_period.index[val_mask]
    names = [f for f in FAMILY_ORDER if f in runs]
    D = np.column_stack([runs[f]["daily"].to_numpy()[val_mask] - m0["daily"].to_numpy()[val_mask] for f in names])
    spa = stats.spa_test(D, gb["paired_bootstrap"]["block_sessions"], PREREG["overfitting"]["spa_reps"],
                         gb["paired_bootstrap"]["seed"])
    spa["models"] = names

    # --- 4. PBO và DSR trên toàn bộ cấu hình đã thử (S6 + S8), dev+val
    m6 = ctx.load("S6", "daily_dev_val").set_index("day").drop(columns=["A0_KE_goc"], errors="ignore")
    m8 = ctx.load("S8", "daily_dev_val").set_index("day")
    trials = pd.concat([m6, m8.drop(columns=[c for c in m8.columns if c in m6.columns])], axis=1).dropna()
    pbo = stats.pbo_cscv(trials.to_numpy(), PREREG["overfitting"]["pbo_blocks"])
    in_val = trials.index.isin(val_idx)
    sr_val_trials = np.array([daily_sr(trials.loc[in_val, c].to_numpy()) for c in trials.columns])
    dsr = {f: dsr_for(runs[f]["daily"].to_numpy()[val_mask], sr_val_trials) for f in names}
    dsr["M0"] = dsr_for(m0["daily"].to_numpy()[val_mask], sr_val_trials)

    result = {"gate_table": rows, "passed": passed, "M9": m9, "SPA_val": spa,
              "PBO_dev_val": {**pbo, "trials": int(trials.shape[1])}, "DSR_val": dsr, "egarch_gate": gate,
              "locked": locked}

    # --- 5. tập test: một lần, có khóa
    lock_path = os.path.join(out, "TEST_LOCK.json")
    if os.path.exists(lock_path) and not getattr(ctx, "allow_test_rerun", False):
        test = {"status": "BỎ QUA: TEST_LOCK.json đã tồn tại (test đã chạy một lần). "
                          "Dùng allow_test_rerun=True để chạy lại; việc chạy lại sẽ được ghi nhận.",
                "previous": read_json(lock_path).get("timestamp")}
    else:
        rerun = os.path.exists(lock_path)
        locked_runs = {"M0": m0}
        for k, spec in locked.items():
            if k != "M0":
                locked_runs[k] = ex.run(spec)
        test = {"models": {}, "cost_sensitivity": {}}
        tmask = (ex.day_period == "test").to_numpy()
        for i, (k, r) in enumerate(locked_runs.items()):
            entry = {"metrics": ex.metrics(r, periods=("test",))["test"]}
            if k != "M0":
                c = compare(ex, r, m0, "test", gb["paired_bootstrap"]["seed"] + 200 + i)
                entry["vs_M0"] = {"dU": c["dU"], "p_one_sided": c["p_one_sided"], "ci95": c["ci95"],
                                  "mdd_ratio": c["mdd_ratio"]}
            entry["DSR"] = dsr_for(r["daily"].to_numpy()[tmask], sr_val_trials)
            test["models"][k] = entry
        others = [k for k in locked_runs if k != "M0"]
        if others:
            Dt = np.column_stack([locked_runs[k]["daily"].to_numpy()[tmask] - m0["daily"].to_numpy()[tmask] for k in others])
            test["SPA"] = {**stats.spa_test(Dt, gb["paired_bootstrap"]["block_sessions"],
                                            PREREG["overfitting"]["spa_reps"], 7), "models": others}
        for tag, kw in {"tax0": {"tax_rate": 0.0}, "tax0.2pct": {"tax_rate": 0.002},
                        "slip0.1": {"slippage_points": 0.1}, "slip0.4": {"slippage_points": 0.4}}.items():
            ex2 = Experiment(lab)
            ex2.cfg = runner.make_config(**{"tax_rate": PREREG["costs"]["tax_rate"], **kw})
            res = {k: ex2.metrics(ex2.run(spec), periods=("val", "test")) for k, spec in locked.items()}
            test["cost_sensitivity"][tag] = {k: {p: {"U": v["U"], "net_points": v["net_points"]} for p, v in d.items()}
                                              for k, d in res.items()}
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        test["timestamp"], test["rerun"] = stamp, bool(rerun)
        write_json(lock_path, {"timestamp": stamp, "rerun": bool(rerun), "locked": list(locked),
                               "prereg_sha256": prereg.digest(),
                               "spec_sha256": hashlib.sha256(json.dumps(locked, sort_keys=True).encode()).hexdigest()})
        write_json(os.path.join(out, "test_results.json"), test)
    result["test"] = test
    write_json(os.path.join(out, "final_results.json"), result)
    write_report(ctx, result)
    return {"passed": passed, "M9": ("đã dựng" if "spec" in m9 else "không dựng"),
            "SPA_p": round(spa["p_value"], 4), "PBO": round(pbo["pbo"], 3),
            "test": ("đã chạy" if test.get("models") else test.get("status", ""))}


# ------------------------------------------------------------------------ báo cáo
def write_report(ctx, result: dict) -> str:
    R = os.path.join(ctx.root, "results")

    def rj(step, name):
        p = os.path.join(R, step, name)
        return read_json(p) if os.path.exists(p) else None

    s0, s1, s2, s3 = rj("S0", "audit_report.json"), rj("S1", "summary.json"), rj("S2", "report.json"), rj("S3", "report.json")
    s5, s7, s8 = rj("S5", "predictive.json"), rj("S7", "report.json"), rj("S8", "selection.json")
    tab6 = ctx.load("S6", "ablation_table")
    tab8 = ctx.load("S8", "models_table")
    L = ["# Báo cáo thí nghiệm KE Recovery (VN30F1M)", "",
         f"*Tạo tự động; mã băm đăng ký trước: `{prereg.digest()[:16]}`. Mọi kết quả là điểm P&L ròng sau phí, trượt giá và thuế 0,1%, 1 hợp đồng.*", ""]
    m0v = tab8[tab8["config"] == "M0"].iloc[0]
    gate_txt = "QUA" if result["egarch_gate"] else "KHÔNG QUA"
    if s7:
        ev = s7["evaluation"]["val"]
        gate_txt += (f" (QLIKE val {ev[s7['chosen_distribution']]['qlike']:.4f} so với đối chứng tốt nhất "
                     f"{s7['best_benchmark_val']} {ev['benchmarks'][s7['best_benchmark_val']]['qlike']:.4f}, "
                     f"DM p = {s7['dm_vs_best_benchmark_val']['p_a_better']:.3f})")
    L += ["## 1. Tóm tắt", "",
          f"- **M0 (KE gốc)**: U (Sharpe ngày) dev {m0v['dev_U']:.2f}, val {m0v['val_U']:.2f}; lợi nhuận ròng dev {m0v['dev_net_points']:.0f}, val {m0v['val_net_points']:.0f} điểm.",
          f"- **Cổng EGARCH-X**: {gate_txt}.",
          f"- **Cải tiến qua cổng trên val**: {', '.join(result['passed']) if result['passed'] else 'KHÔNG CÓ'}.",
          f"- **M9**: {result['M9'].get('note', 'đã dựng; qua cổng = ' + str(result['M9'].get('passed')))}.",
          f"- **PBO** (dev+val, {result['PBO_dev_val']['trials']} cấu hình) = {result['PBO_dev_val']['pbo']:.2f}; **SPA val** p = {result['SPA_val']['p_value']:.3f}.", ""]
    if result["test"].get("models"):
        t = result["test"]["models"]
        L += ["- **Tập test (chạy một lần)**: " + "; ".join(
            f"{k}: U {v['metrics']['U']:.2f}, {v['metrics']['net_points']:.0f} điểm" for k, v in t.items()) + ".", ""]
    else:
        L += [f"- **Tập test**: {result['test'].get('status', 'chưa chạy')}", ""]
    if s0:
        a = s0["audit_1m"]
        L += ["## 2. Dữ liệu và kiểm toán (S0-S2)", "",
              f"- {a['rows']:,} nến 1 phút, {a['sessions']} phiên ({a['first_ts'][:10]} - {a['last_ts'][:10]}); nến chưa phân loại mua/bán: "
              f"{a['unclassified_rows']:,} ({a['unclassified_volume_share']:.2%} khối lượng).",
              f"- 5m gộp từ 1m khớp giá file 5m gốc: {s0['matches_old_5m_prices']}; lệch khối lượng ở {s0['volume_mismatch_bars_vs_old_5m']} nến."]
    if s1:
        p0 = s1["p0_old5m_tax0"]
        L += [f"- Tái lập báo cáo gốc (thuế 0): {p0['trades']} lệnh, {p0['net_points']:.1f} điểm, profit factor {p0['profit_factor']:.2f}."]
    if s2:
        L += [f"- {s2['rolls']} lần roll; Delta ước lượng bằng chỉ số VN30 (xấp xỉ): trung bình {s2['delta_mean']:.2f}, độ lệch chuẩn {s2['delta_std']:.2f} điểm; "
              f"lần roll bất thường: {', '.join(s2['outlier_rolls']) or 'không'}.",
              f"- Độ lệch chuẩn lợi suất qua đêm ngày roll: {s2['r_night_std_roll_days_raw']:.4f} (giá thực) -> {s2['r_night_std_roll_days_adj']:.4f} (đã hiệu chỉnh).", ""]
    if s3:
        L += ["## 3. Đặc trưng (S3)", "",
              f"- ATC: độ lệch chuẩn I = {s3['I_std_atc']:.2f} so với {s3['I_std_non_atc']:.2f} ở nến thường; sau chuẩn hóa theo ô "
              f"{s3['I_tilde_std_atc']:.2f} so với {s3['I_tilde_std_non_atc']:.2f}.", ""]
    if s5:
        h6 = s5["h6"]
        rows5 = [{"test": r["test"], "period": r["period"], "p": r["p"], "p_holm": r.get("p_holm", "")}
                 for r in h6["groups"] + h6["calendar"] if r["test"] != "H3_signed_extra"]
        L += ["## 4. Kiểm định giả thuyết (S5, chân trời 6 nến)", "",
              md_table(rows5, ["test", "period", "p", "p_holm"], {"p": "{:.4f}", "p_holm": "{:.3f}"}),
              "IC (Spearman, trung bình theo phiên): " + "; ".join(
                  f"{r['var']} {r['period']}: {r['mean_ic']:.3f} (t={r['t_across_days']:.1f})" for r in h6["ic"]), ""]
        if s7 and s7.get("H5"):
            L += ["H5 (phụ thuộc phương sai dự báo): " + "; ".join(f"{r['period']}: p={r['p']:.3f}" for r in s7["H5"]), ""]
    cols6 = ["config", "dev_U", "val_U", "dev_net_points", "val_net_points", "dev_trades"]
    L += ["## 5. Ablation M0 (S6)", "",
          md_table(tab6[cols6].to_dict("records"), cols6,
                   {"dev_U": "{:.2f}", "val_U": "{:.2f}", "dev_net_points": "{:.0f}", "val_net_points": "{:.0f}"})]
    if s7:
        ev = s7["evaluation"]
        L += ["## 6. EGARCH-X (S7)", "", f"Phân phối chọn theo log-score val: **{s7['chosen_distribution']}**.", "",
              md_table([{"dist": k, "val_qlike": v["qlike"], "val_log_score": v["log_score"], "val_pit_ks_p": v["pit_ks_p"],
                         "val_var01_rate": v["var01"]["rate"], "val_var05_rate": v["var05"]["rate"]}
                        for k, v in ev["val"].items() if k != "benchmarks"],
                       ["dist", "val_qlike", "val_log_score", "val_pit_ks_p", "val_var01_rate", "val_var05_rate"],
                       {c: "{:.4f}" for c in ("val_qlike", "val_log_score", "val_pit_ks_p", "val_var01_rate", "val_var05_rate")}),
              "Đối chứng (QLIKE val): " + "; ".join(f"{k} {v['qlike']:.4f}" for k, v in ev["val"]["benchmarks"].items()), ""]
    L += ["## 7. M1-M9: cổng trên val (S8-S9)", "",
          md_table(result["gate_table"], ["family", "config", "eligible", "dU_val", "p", "p_holm", "mdd_ratio", "passed"],
                   {c: "{:.3f}" for c in ("dU_val", "p", "p_holm", "mdd_ratio")}), ""]
    if "spec" in result["M9"]:
        L += [f"M9 = {'+'.join(result['M9']['components'])}: dU val {result['M9']['dU_val']:.3f}, p {result['M9']['p']:.3f}, "
              f"qua cổng: {result['M9']['passed']}.", ""]
    if s8:
        L += ["Chẩn đoán S8: `" + json.dumps(s8["diagnostics"], ensure_ascii=False, default=str)[:1500] + "`", ""]
    L += ["## 8. Chống quá khớp", "",
          f"- PBO = {result['PBO_dev_val']['pbo']:.2f} trên {result['PBO_dev_val']['trials']} cấu hình ({result['PBO_dev_val']['combos']} tổ hợp CSCV).",
          f"- SPA (val, {len(result['SPA_val']['models'])} mô hình so với M0): t = {result['SPA_val']['t_stat']:.2f}, p = {result['SPA_val']['p_value']:.3f}.",
          "- DSR (val): " + "; ".join(f"{k}: {v['dsr']:.2f}" for k, v in result["DSR_val"].items()), ""]
    if result["test"].get("models"):
        L += ["## 9. Tập test (chạy một lần, có khóa)", "",
              md_table([{"model": k, "U": v["metrics"]["U"], "net_points": v["metrics"]["net_points"],
                         "mdd": v["metrics"]["mdd_points"], "trades": v["metrics"]["trades"],
                         "dU_vs_M0": v.get("vs_M0", {}).get("dU", ""), "p": v.get("vs_M0", {}).get("p_one_sided", ""),
                         "DSR": v["DSR"]["dsr"]} for k, v in result["test"]["models"].items()],
                       ["model", "U", "net_points", "mdd", "trades", "dU_vs_M0", "p", "DSR"],
                       {**{c: "{:.3f}" for c in ("U", "dU_vs_M0", "p", "DSR")}, "net_points": "{:.1f}", "mdd": "{:.1f}"}), ""]
    L += ["## 10. Giới hạn và điều cần thừa nhận", "",
          "- Delta của lần roll là ước lượng bằng chỉ số VN30 (basis), không đo tại thời điểm chồng lấn như đề cương mô tả; có một lần roll bất thường (21/05/2020).",
          "- Thuế 0,1% áp cho toàn lịch sử với ký quỹ 18,48%; Thông tư 87/2026 chỉ hiệu lực từ 01/07/2026 và tỷ lệ VSDC thật thay đổi theo ngày. Phí 0,005% được coi là all-in, không cộng phí hạ tầng.",
          "- M0 đã được xem trên tập test ở bước S1 (benchmark của mã đóng băng); không dùng để chọn gì.",
          "- Các tham số đề cương chưa ấn định (cửa sổ, epsilon, lưới lambda, ngưỡng, luật chế độ M8...) là lựa chọn của nhóm thực hiện, ghi trong preregistration.json; hai tham số (eps0 và luật M8) được chọn lại sau khi thấy lỗi thang đo/giao cắt giả, trước mọi kết quả giao dịch.",
          "- M4 học trên lệnh M0 thật nhưng áp cho mọi nến ứng viên (xấp xỉ). Nến ATC không bị loại ở cấu hình chính (độ nhạy loại ATC là bổ sung của nhóm thực hiện).",
          "- Không phân biệt được nến thiếu do 'không khớp lệnh' hay 'lỗi thu thập' khi chỉ có OHLCV.", ""]
    path = os.path.join(ctx.out("S9"), "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path
