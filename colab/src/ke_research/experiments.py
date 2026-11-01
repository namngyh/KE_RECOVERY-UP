"""Bộ chạy thí nghiệm: spec -> tín hiệu -> engine QPT -> P&L ngày -> chỉ số U theo giai đoạn.

Quy ước:
* Mọi cấu hình chạy trên CÙNG engine, 1 hợp đồng, chi phí và thuế trong prereg (runner.make_config).
* P&L ngày (điểm) = hiệu đường vốn (mark-to-market, sau phí/trượt giá/thuế) ở nến cuối mỗi phiên, chia hệ số nhân.
* U = Sharpe ngày thường niên hóa sqrt(250) * mean / std (đề cương mục 11.3).
* Tập test được niêm phong: metrics(..., periods) chỉ trả các giai đoạn được yêu cầu; các bước chọn lựa chỉ xin dev/val.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import runner, stats
from .models import Lab, build_signal
from .prereg import PREREG, period_of

PPY = PREREG["utility"]["periods_per_year"]


class Experiment:
    def __init__(self, lab: Lab, tax_rate: float | None = None):
        self.lab = lab
        tax = PREREG["costs"]["tax_rate"] if tax_rate is None else tax_rate
        self.cfg = runner.make_config(tax_rate=tax)
        self.bars = runner.engine_frame(lab.raw[["ts", "open_time_ms", "open", "high", "low", "close"]]).copy()
        last = lab.raw.groupby(lab.day).apply(lambda x: x.index[-1], include_groups=False)
        self.days = pd.DatetimeIndex(last.index)
        self.last_idx = last.to_numpy()
        self.day_period = pd.Series(lab.period[self.last_idx], index=self.days)
        self._runs: dict = {}

    @staticmethod
    def key(spec: dict) -> str:
        return json.dumps(spec, sort_keys=True, default=str)

    def run(self, spec: dict | None = None, signal: np.ndarray | None = None) -> dict:
        spec = spec or {}
        k = self.key(spec) if signal is None else None
        if k is not None and k in self._runs:
            return self._runs[k]
        sig = build_signal(self.lab, spec) if signal is None else np.asarray(signal)
        res = runner.run(self.bars, sig, self.cfg)
        eq_last = np.asarray(res.equity, float)[self.last_idx]
        pnl = np.diff(np.concatenate([[self.cfg.initial_capital], eq_last])) / runner.MULTIPLIER
        out = {"signal": sig, "daily": pd.Series(pnl, index=self.days), "trades": runner.trades_frame(res),
               "result": res, "spec": spec}
        if k is not None:
            self._runs[k] = out
        return out

    def metrics(self, run: dict, periods=("dev", "val")) -> dict:
        daily = run["daily"]
        trades = run["trades"]
        tp = period_of(trades["entry_time"]) if len(trades) else pd.Series([], dtype=object)
        out = {}
        for p in periods:
            d = daily[self.day_period == p]
            tr = trades[tp == p] if len(trades) else trades
            net = tr["net_points"] if len(tr) else pd.Series(dtype=float)
            out[p] = {
                "U": float(stats.sharpe(d.to_numpy(), PPY)) if len(d) > 2 and d.std() > 0 else float("nan"),
                "net_points": float(d.sum()), "mean_day": float(d.mean()) if len(d) else float("nan"),
                "mdd_points": stats.max_drawdown(d.to_numpy()), "days": int(len(d)),
                "trades": int(len(tr)), "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
                "profit_factor": float(net[net > 0].sum() / -net[net < 0].sum()) if (net < 0).any() else float("inf"),
            }
        return out

    def daily_slice(self, run: dict, period: str) -> pd.Series:
        return run["daily"][self.day_period == period]

    def matrix(self, runs: dict[str, dict], periods=("dev", "val")) -> pd.DataFrame:
        """Ma trận P&L ngày (hàng = phiên, cột = cấu hình) cho PBO/SPA, chỉ gồm các giai đoạn được phép."""
        keep = self.day_period.isin(periods)
        return pd.DataFrame({n: r["daily"][keep] for n, r in runs.items()})
