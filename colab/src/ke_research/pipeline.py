"""Chạy toàn bộ thí nghiệm một lượt: S0 -> S1 -> (các bước sau được thêm vào STEPS).

Mỗi bước:
* đọc đầu vào từ <root>/data và kết quả bước trước trong <root>/results/<bước>;
* ghi kết quả vào <root>/results/<bước>/ cùng DONE.json (đánh dấu đã xong);
* có "cổng" (assert): sai là dừng cả pipeline, không chạy tiếp bước sau.
Bước đã có DONE.json được bỏ qua trừ khi force=True. Thêm bước mới: viết hàm step_xx(ctx) rồi
đăng ký vào STEPS.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field

import pandas as pd

import numpy as np

from . import calendar_feats, data, rolls, runner

KE_SHA256 = "487a835492e6c9b6b5a4aa8e468aedc397755b183970dfa18fda338e8ee478f3"
OLD_5M = "VN30F1M_5m_20171106-20261030.csv"
ONE_MIN = "ohlc_export.csv"
INDEX_1D = "VN30_1d_20170103-20261030.csv"
# S1: số liệu báo cáo gốc phải tái lập được
P0_TRADES, P0_WINS, P0_NET_POINTS = 1867, 890, 4337.4
TAX_RATE = 0.001  # 0,1%


@dataclass
class Context:
    root: str
    fmt: str = "parquet"            # "parquet" (Colab) hoặc "csv" (máy không có pyarrow)
    data_dir: str | None = None
    allow_test_rerun: bool = False   # True: cho phép chạy lại tập test dù đã có TEST_LOCK.json (sẽ được ghi nhận)
    log: list = field(default_factory=list)

    @property
    def data(self) -> str:
        return self.data_dir or os.path.join(self.root, "data")

    def out(self, step: str) -> str:
        path = os.path.join(self.root, "results", step)
        os.makedirs(path, exist_ok=True)
        return path

    def save(self, df: pd.DataFrame, step: str, name: str) -> str:
        path = os.path.join(self.out(step), f"{name}.{'parquet' if self.fmt == 'parquet' else 'csv'}")
        df.to_parquet(path, index=False) if self.fmt == "parquet" else df.to_csv(path, index=False)
        return path

    def exists(self, step: str, name: str) -> bool:
        ext = "parquet" if self.fmt == "parquet" else "csv"
        return os.path.exists(os.path.join(self.root, "results", step, f"{name}.{ext}"))

    def load(self, step: str, name: str) -> pd.DataFrame:
        base = os.path.join(self.root, "results", step, name)
        if self.fmt == "parquet":
            return pd.read_parquet(base + ".parquet")
        df = pd.read_csv(base + ".csv")
        for c in ("ts", "day", "exp_day", "new_day", "entry_time", "exit_time", "first_ts", "last_ts"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        return df


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


def _ke_params():
    from strategies import ke_recovery as ke
    return ke, {k: v["default"] for k, v in ke.STRATEGY["params"].items()}


def _check_frozen(ctx: Context) -> None:
    import strategies.ke_recovery as ke
    sha = hashlib.sha256(open(ke.__file__, "rb").read()).hexdigest()
    if sha != KE_SHA256:
        raise AssertionError("ke_recovery.py đã bị sửa so với bản đóng băng")


# ------------------------------------------------------------------------ S0
def step_s0(ctx: Context) -> dict:
    df1 = data.load_1m(os.path.join(ctx.data, ONE_MIN))
    audit = data.audit_1m(df1)
    df5 = data.resample(df1, "5min")
    checks = data.check_resample(df1, df5)
    conserved = all(v for k, v in checks.items() if k.endswith("_equal") or k == "minutes_conserved")
    if not conserved:
        raise AssertionError(f"Gộp 5m làm mất số liệu: {checks}")
    for key in ("duplicate_ts", "null_cells", "non_positive_price", "high_below_open_or_close",
                "low_above_open_or_close", "negative_volume", "classified_gt_volume"):
        if audit[key] != 0:
            raise AssertionError(f"Kiểm toán 1m: {key} = {audit[key]}")
    # 5m gộp phải khớp file 5m gốc về giá trên khoảng chung
    old = pd.read_csv(os.path.join(ctx.data, OLD_5M))
    old["ts"] = pd.to_datetime(old["datetime_vn"])
    merged = old.merge(df5, on="ts", suffixes=("_o", "_n"))
    common_ok = len(merged) == len(df5) and all(
        (merged[f"{c}_o"] - merged[f"{c}_n"]).abs().max() < 1e-9 for c in ("open", "high", "low", "close"))
    if not common_ok:
        raise AssertionError("5m gộp từ 1m không khớp giá với file 5m gốc")
    ctx.save(df1, "S0", "bars_1m")
    ctx.save(df5, "S0", "bars_5m")
    out = ctx.out("S0")
    _write_json(os.path.join(out, "audit_report.json"),
                {"audit_1m": audit, "resample_checks": checks, "matches_old_5m_prices": common_ok,
                 "volume_mismatch_bars_vs_old_5m": int(((merged["volume_o"] - merged["volume_n"]).abs() > 0.5).sum())})
    return {"rows_1m": audit["rows"], "bars_5m": checks["bars_5m"], "unclassified_rows": audit["unclassified_rows"],
            "matches_old_5m_prices": common_ok}


# ------------------------------------------------------------------------ S1
def step_s1(ctx: Context) -> dict:
    _check_frozen(ctx)
    ke, params = _ke_params()
    old = pd.read_csv(os.path.join(ctx.data, OLD_5M))
    sig_old = ke.signals(old, params).to_numpy()

    p0 = runner.summarize(runner.run(old, sig_old, runner.make_config(tax_rate=0.0)))
    if not (p0["trades"] == P0_TRADES and p0["wins"] == P0_WINS and abs(p0["net_points"] - P0_NET_POINTS) < 0.05):
        raise AssertionError(f"P0 không tái lập được báo cáo gốc: {p0}")

    res_tax = runner.run(old, sig_old, runner.make_config(tax_rate=TAX_RATE))
    tax_full = runner.summarize(res_tax)
    ctx.save(runner.trades_frame(res_tax), "S1", "trades_M0_old5m_tax0p1")

    b5 = ctx.load("S0", "bars_5m")
    res_new = runner.run(b5, ke.signals(b5, params).to_numpy(), runner.make_config(tax_rate=TAX_RATE))
    new = runner.summarize(res_new)
    trades = runner.trades_frame(res_new)
    ctx.save(trades, "S1", "trades_M0_5mFrom1m_tax0p1")
    _write_json(os.path.join(ctx.out("S1"), "summary.json"),
                {"p0_old5m_tax0": p0, "old5m_tax0p1": tax_full, "new5m_to_20260717_tax0p1": new, "params": params})
    return {"p0": {k: p0[k] for k in ("trades", "wins", "net_points")},
            "net_points_tax0p1_full": round(tax_full["net_points"], 1),
            "net_points_tax0p1_to_0717": round(new["net_points"], 1), "trades_to_0717": new["trades"]}


# ------------------------------------------------------------------------ S2
def step_s2(ctx: Context) -> dict:
    """Roll, chuỗi giá hiệu chỉnh nhân quả, lợi suất trong phiên/qua đêm, biến lịch (đề cương mục 5)."""
    _check_frozen(ctx)
    ke, params = _ke_params()
    b5 = ctx.load("S0", "bars_5m")
    index_close = rolls.load_index_daily(os.path.join(ctx.data, INDEX_1D))
    rt = rolls.roll_table(b5, index_close)
    if rt.empty:
        raise AssertionError("Không tìm thấy lần roll nào")
    if not np.isfinite(rt[["delta", "delta_robust"]].to_numpy()).all():
        raise AssertionError("Delta không hữu hạn")
    adj = rolls.forward_adjust(b5, rt, "delta")
    adj_robust = rolls.forward_adjust(b5, rt, "delta_robust")
    if not (adj[rolls.PRICE_COLS] > 0).all().all():
        raise AssertionError("Chuỗi hiệu chỉnh có giá không dương (mục D1)")

    # Cổng nhân quả: chỉ báo KE trên chuỗi thuận == chuỗi cộng lùi dùng chỉ các roll đã xảy ra (mục 5.2)
    ind_f = ke.indicators(adj, params)
    rng = np.random.default_rng(0)
    cuts = sorted({int(rt["roll_idx"].iloc[k]) + off for k in (20, 50, 80) for off in (0, 1, 40)}
                  | set(int(c) for c in rng.integers(3000, len(b5), 4)))
    for t in cuts:
        pre = rolls.asof_back_adjusted(b5, rt, t, "delta")
        ind_a = ke.indicators(pre, params)
        for col in ("f", "s"):
            if abs(ind_f[col].iloc[t] - ind_a[col].iloc[t]) > 1e-9:
                raise AssertionError(f"Chuỗi thuận khác chuỗi nhân quả ở nến {t}: {col}")
        if (adj["close"].iloc[t] > ind_f["ema"].iloc[t]) != (pre["close"].iloc[t] > ind_a["ema"].iloc[t]):
            raise AssertionError(f"Cổng EMA khác nhau ở nến {t}")

    ret = rolls.intraday_overnight_returns(adj)
    cal = calendar_feats.session_calendar(b5, rt)
    gaps = calendar_feats.classify_bar_gaps(b5)
    bar_cal = calendar_feats.add_bar_calendar(b5, cal)

    ctx.save(rt, "S2", "roll_table")
    ctx.save(adj.assign(**{c: bar_cal[c] for c in ("slot", "bar_of_day")}), "S2", "bars_5m_adj")
    ctx.save(adj_robust[["ts", "cum_offset"]].rename(columns={"cum_offset": "cum_offset_robust"}), "S2", "offset_robust")
    ctx.save(ret.reset_index(), "S2", "session_returns")
    ctx.save(cal.reset_index(), "S2", "session_calendar")
    ctx.save(gaps["per_session"].reset_index(), "S2", "bar_gaps_per_session")
    roll_days = rt["new_day"]
    summary = {
        "rolls": int(len(rt)),
        "delta_mean": float(rt["delta"].mean()), "delta_std": float(rt["delta"].std()),
        "delta_abs_max": float(rt["delta"].abs().max()),
        "outlier_rolls": [str(d.date()) for d in rt.loc[rt["outlier"], "exp_day"]],
        "adj_price_min": float(adj[rolls.PRICE_COLS].min().min()),
        "cum_offset_range": [float(adj["cum_offset"].min()), float(adj["cum_offset"].max())],
        "r_night_std_roll_days_raw": float(ret.loc[roll_days, "r_night_raw"].std()),
        "r_night_std_roll_days_adj": float(ret.loc[roll_days, "r_night"].std()),
        "causal_equivalence_cuts_checked": len(cuts),
        "sessions": int(len(cal)), "near_expiry_sessions": int(cal["near_expiry"].sum()),
        "after_holiday_sessions": int(cal["after_holiday"].sum()),
        "k_d_estimated_sessions": int(cal["k_estimated"].sum()),
        "bar_gaps": gaps["summary"],
        "delta_method": "basis (chỉ số VN30): [F-I](ngày sau roll) - [F-I](ngày đáo hạn); xấp xỉ, xem rolls.py",
    }
    _write_json(os.path.join(ctx.out("S2"), "report.json"), summary)
    return {k: summary[k] for k in ("rolls", "outlier_rolls", "delta_mean", "delta_std",
                                   "r_night_std_roll_days_raw", "r_night_std_roll_days_adj",
                                   "causal_equivalence_cuts_checked", "near_expiry_sessions")}


from . import steps as _steps, steps_final as _final  # noqa: E402  (S3-S9; cần Context/ke_research đã nạp)

STEPS = [("S0", step_s0), ("S1", step_s1), ("S2", step_s2), ("S3", _steps.step_s3), ("S4", _steps.step_s4),
         ("S5", _steps.step_s5), ("S6", _steps.step_s6), ("S7", _steps.step_s7), ("S8", _final.step_s8),
         ("S9", _final.step_s9)]


# --------------------------------------------------------------------- runner
def run_all(root: str, only: list[str] | None = None, force: bool = False, fmt: str = "parquet",
            data_dir: str | None = None, allow_test_rerun: bool = False) -> list[dict]:
    """Chạy các bước theo thứ tự; dừng ở bước đầu tiên lỗi. Trả về nhật ký từng bước."""
    ctx = Context(root=root, fmt=fmt, data_dir=data_dir, allow_test_rerun=allow_test_rerun)
    for p in (root, os.path.join(root, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    log_path = os.path.join(root, "results", "run_log.json")
    for name, fn in STEPS:
        if only and name not in only:
            continue
        done = os.path.join(ctx.out(name), "DONE.json")
        entry = {"step": name}
        if os.path.exists(done) and not force:
            entry.update(status="skipped (đã có DONE.json; dùng force=True để chạy lại)")
            ctx.log.append(entry); print(f"[{name}] bỏ qua - đã xong")
            continue
        t0 = time.time()
        print(f"[{name}] đang chạy ...")
        try:
            summary = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - cần ghi lại mọi lỗi cổng rồi dừng
            entry.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                         seconds=round(time.time() - t0, 1))
            ctx.log.append(entry)
            _write_json(log_path, ctx.log)
            print(f"[{name}] LỖI: {entry['error']}")
            traceback.print_exc()
            return ctx.log
        entry.update(status="OK", seconds=round(time.time() - t0, 1), summary=summary)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        with open(os.path.join(ctx.out(name), "pip_freeze.txt"), "w") as f:
            f.write(freeze)
        _write_json(done, entry)
        ctx.log.append(entry)
        _write_json(log_path, ctx.log)
        print(f"[{name}] xong trong {entry['seconds']}s: {json.dumps(summary, ensure_ascii=False)}")
    return ctx.log
