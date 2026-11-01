"""Chạy một chiến lược/tín hiệu qua engine QPT với quy ước cố định của nghiên cứu.

Quy ước (đề cương mục 9, khớp báo cáo backtest gốc):
* 1 hợp đồng cố định, hệ số nhân 100.000 đ/điểm, vốn 100 triệu đồng.
* Phí 0,005%/chiều trên giá trị danh nghĩa; trượt giá 0,2 điểm/chiều.
* Ký quỹ mô phỏng 18,48%; không giải chấp (force_sale="none"), ngưỡng 17,35% chỉ để đánh dấu.
* Đáo hạn: cắt lệnh tại giá đóng cửa nến cuối của hợp đồng cũ và không vào lại tới khi
  tín hiệu đổi (expiry="close"). Đây là cấu hình tái lập được 1.867 lệnh / +4.337,4 điểm.
* Khớp ở giá mở nến kế tiếp (fill_at="open").
* tax_rate là PHÂN SỐ: 0,001 = 0,1%. Thuế mỗi chiều = tax_rate x giá x 100.000 x ký quỹ / 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.strategy.engine import BacktestConfig, BacktestResult, run_backtest
from backend.strategy.position_model import ContractConfig

MULTIPLIER = 100_000.0
INITIAL_MARGIN = 0.1848
FORCE_SELL_RATE = 0.1735
CAPITAL = 100_000_000.0
FEE_RATE = 0.00005
SLIPPAGE_POINTS = 0.2


def make_config(tax_rate: float = 0.0, expiry: str = "close", slippage_points: float = SLIPPAGE_POINTS,
                fee_rate: float = FEE_RATE, capital: float = CAPITAL) -> BacktestConfig:
    contract = ContractConfig(
        initial_margin_rate=INITIAL_MARGIN,
        maintenance_threshold=FORCE_SELL_RATE / INITIAL_MARGIN,
        sizing="fixed", contracts=1, max_contracts=1,
        multiplier=MULTIPLIER,
        fee_mode="notional", fee_rate=fee_rate,
        slippage_points=slippage_points,
        series_label="F1M", force_sale="none", expiry=expiry,
        tax_rate=tax_rate,
    )
    return BacktestConfig(initial_capital=capital, contract=contract, fill_at="open")


def engine_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên open_time_ms -> open_time (cột engine yêu cầu); giữ nguyên các cột khác."""
    if "open_time" in bars.columns:
        return bars
    return bars.rename(columns={"open_time_ms": "open_time"})


def run(bars: pd.DataFrame, signal, config: BacktestConfig | None = None) -> BacktestResult:
    df = engine_frame(bars).reset_index(drop=True)
    sig = np.asarray(signal)
    if len(sig) != len(df):
        raise ValueError(f"signal dài {len(sig)} nhưng dữ liệu có {len(df)} nến")
    return run_backtest(df, sig.astype("int8"), config or make_config())


def trades_frame(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        net = t.net_pnl if t.net_pnl is not None else t.pnl
        rows.append({
            "side": t.side, "entry_index": t.entry_index, "exit_index": t.exit_index,
            "entry_time": pd.to_datetime(t.entry_time, unit="s") + pd.Timedelta(hours=7),
            "exit_time": pd.to_datetime(t.exit_time, unit="s") + pd.Timedelta(hours=7),
            "entry_price": t.entry_price, "exit_price": t.exit_price, "points": t.points,
            "net_pnl": net, "net_points": net / MULTIPLIER,
            "entry_fee": t.entry_fee, "exit_fee": t.exit_fee,
            "bars_held": t.bars_held, "exit_reason": t.exit_reason,
        })
    return pd.DataFrame(rows)


def summarize(result: BacktestResult) -> dict:
    """Chỉ số tóm tắt theo điểm, sau mọi chi phí đã tính trong engine."""
    tr = trades_frame(result)
    if tr.empty:
        return {"trades": 0}
    pts = tr["net_points"]
    gain, loss = pts[pts > 0].sum(), -pts[pts < 0].sum()
    eq = np.asarray(result.equity, dtype=float) / MULTIPLIER  # điểm
    dd = (np.maximum.accumulate(eq) - eq).max()
    return {
        "trades": int(len(tr)),
        "net_points": float(pts.sum()),
        "gross_points_before_costs": float(tr["points"].sum()),
        "wins": int((pts > 0).sum()),
        "win_rate": float((pts > 0).mean()),
        "profit_factor": float(gain / loss) if loss > 0 else float("inf"),
        "expectancy_points": float(pts.mean()),
        "max_drawdown_points": float(dd),
        "final_equity_vnd": float(result.equity[-1]),
        "long_trades": int((tr["side"] == "long").sum()),
        "short_trades": int((tr["side"] == "short").sum()),
    }
