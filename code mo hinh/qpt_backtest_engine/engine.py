"""The backtest engine.

Execution model — the assumptions every result rests on:

* A signal produced at the close of bar *i* is executed at the **open of bar
  i+1**. Nothing is ever filled at a price the strategy could already see, so
  results carry no look-ahead bias. ``BacktestConfig.fill_at = "close"`` (an
  option, 2026-10-29) fills it at the **close of bar i** instead, after that
  bar's stops and liquidation; that price is one the strategy has seen, so the
  option is optimistic unless the order really goes in as the bar closes.
  ``fill_at = "next_close"`` fills it at the close of bar i+1, by the same rules.
* Fills are worsened by ``slippage`` in the direction that hurts, and both
  entries and exits pay ``fee`` on notional.
* Each position commits ``size_pct`` of current equity as margin and controls
  ``margin * leverage`` of notional.
* A leveraged position is liquidated intrabar when the loss reaches the
  committed margin — checked against the bar's low (long) or high (short),
  so a wick that would have wiped the position out is not skipped over. A bar
  that opens past the forced-close price fills at that open, worsened by
  slippage, so the loss can exceed the margin: no trade happened at the level.
  Index futures on the contract model are the exception (Nam, 2026-10-14):
  the backtest does not force-close them. Sized by % of capital, the first
  bar of each position that reaches the broker's force-sell price is marked
  (``Trade.liquidation_warning``) and the P&L is left as it is, to say the
  sizing is too large for the strategy. A continuous series (VN30F1M …) is
  settled at the close of each expiry it rolls on.
* Only one position at a time; a reversal closes and reopens at the same bar.
* Exits (optional, ``config.exits``; off by default, and off means the loop
  below is exactly the one it replaced). The same rules as paper trading:
  a stop or target reached inside a bar fills at its level; a bar that opens
  past the stop fills at that open, worsened by slippage; a bar that reaches
  both is scored as the stop, because OHLC cannot say which came first and the
  unfavourable guess is the honest one. A level that moves — a trailing stop
  ratcheting, a stop moved to break-even after a partial target — is judged
  from the next bar, since the order of a bar's high and low is unknown. After
  a stop or target closes the position, the strategy's standing signal in the
  same direction is not re-entered until it changes; otherwise a strategy that
  holds "long" would buy back on the next bar and the stop would mean nothing.

* Vietnamese cash-market rules (optional, ``config.vn_cash``; set by the API
  for VN equities, ETFs and covered warrants, off means the loop below is
  exactly the one it replaced): long only; whole lots of 100 shares; shares
  bought in session T can be sold from 13:00 of session T+2 (T+2.5), so on
  daily bars filled at the open the first sale is session T+3; fills rounded
  to the price step against the trader; a 0.1% tax on every sale; and a bar
  locked at the floor (one price, a band's move below the previous session's
  close) cannot sell, one locked at the ceiling cannot buy.

The engine deliberately owns all of this rather than the strategy, so two
strategies are always compared under identical rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np
import pandas as pd

from backend import progress

from backend.strategy.position_model import ContractConfig, Sizing, model_for


@dataclass
class ExitConfig:
    """Stops and targets for a backtest (Nam, 2026-09-25: the paper set minus
    risk-based sizing). Percentages are of the entry price."""

    stop_pct: float | None = None       # stop this far against the entry
    stop_atr: float | None = None       # or k x Wilder ATR(14) of the bars before entry
    target_pct: float | None = None     # target this far in favour
    trail_pct: float | None = None      # trailing stop, ratchets only
    tp_fraction: float = 1.0            # share closed at the target; (0, 1]
    tp_breakeven: bool = False          # after a partial target, stop to entry

    @property
    def active(self) -> bool:
        return any(v for v in (self.stop_pct, self.stop_atr, self.target_pct, self.trail_pct))

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict | None) -> "ExitConfig | None":
        return cls(**data) if data else None


@dataclass
class VnCashRules:
    """The rules of the Vietnamese cash market a backtest must not skip
    (Nam, 2026-09-26: "backtest trung thực kiểu VN").

    Prices of VN equities are quoted in thousands of dong while capital is in
    dong, so a quantity of 1 000 units is one share (``price_unit``).
    """

    lot_shares: int = 100             # HOSE, HNX and UPCOM board lots
    price_unit: float = 1000.0        # dong per quoted unit
    sell_tax: float = 0.001           # personal income tax on the sale value
    settlement_sessions: int = 2      # T+2 ...
    settle_hour_vn: int = 13          # ... shares usable from 13:00 (T+2.5)
    lock_move: float = 0.069          # a one-price bar this far past the reference is locked
    tick: str = "equity"              # "equity" (HOSE steps) | "flat" (0.01: ETFs, warrants) | "hundred" (HNX, UPCOM)
    # "auto": read the exchange from the prices (backend/data/exchange.py) and
    # take its band and price step; "hose" / "hnx" / "upcom" force one.
    exchange: str = "auto"
    # The exchange as resolved (code, band, tick, evidence), once read: a
    # paper session reads it at start and keeps it, so its rules do not
    # change under an open position.
    exchange_info: dict | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict | None) -> "VnCashRules | None":
        return cls(**data) if data else None


def vn_tick(price: float, table: str = "equity") -> float:
    """Price step in thousands of dong. HOSE shares: 10 đ under 10 000 đ,
    50 đ under 50 000 đ, 100 đ above. ETFs and covered warrants: 10 đ. HNX
    and UPCOM shares: 100 đ at every price."""
    if table == "hundred":
        return 0.1
    if table == "flat" or price < 10.0:
        return 0.01
    return 0.05 if price < 50.0 else 0.1


def vn_round(price: float, direction: int, table: str = "equity") -> float:
    """``price`` on the price grid, against the trader: up when buying
    (``direction`` +1), down when selling."""
    tick = vn_tick(price, table)
    steps = price / tick
    steps = math.ceil(steps - 1e-9) if direction > 0 else math.floor(steps + 1e-9)
    return round(steps * tick, 6)


ATR_LENGTH = 14


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               length: int = ATR_LENGTH) -> np.ndarray:
    """ATR at each bar from that bar and the ones before it; NaN until
    ``length`` bars of true range exist. Same definition as the paper engine."""
    prev = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.vstack([high - low, np.abs(high - prev), np.abs(low - prev)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1 / length, adjust=False).mean().to_numpy(copy=True)
    atr[:length] = np.nan
    return atr


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    size_pct: float = 1.0       # fraction of equity committed as margin
    leverage: float = 1.0
    fee: float = 0.0004         # taker, per side, on notional
    slippage: float = 0.0002    # adverse price move per fill
    # Index futures only: whole contracts, dong per point, margin by rate.
    # None means the linear model (backend/strategy/position_model.py).
    contract: ContractConfig | None = None
    # Stops and targets; None (or all fields empty) means signal exits only.
    exits: ExitConfig | None = None
    # Vietnamese cash-market rules; None means none (crypto, indices, futures).
    vn_cash: VnCashRules | None = None
    # Perpetual funding, as a fraction of the position's value per 8 hours: positive means
    # longs pay and shorts receive. One flat average chosen by the user (Nam, 2026-10-07),
    # not the exchange's history. Applies to the linear (crypto) model only.
    funding_rate: float = 0.0
    # Interest on a borrowed short, as a fraction of the position's value per year; charged only
    # while short, linear (crypto) model only. Booked in the same ``funding`` field as funding.
    borrow_rate: float = 0.0
    # Linear model only: the same quantity on every trade (coins, shares) instead of size_pct of equity (Nam,
    # 2026-10-17: "hợp đồng cố định hay hợp đồng theo vốn"). None means size_pct. Futures size by ContractConfig.
    fixed_quantity: float | None = None
    # Where a signal fills (Nam, 2026-10-29: "chọn giá vào là open hay close"). "open": the signal of bar i fills at the
    # open of bar i+1, the default and the rule of CLAUDE.md §3.1. "close": it fills at the close of bar i itself, as a
    # trader who sends the order the moment the bar closes (or at the ATC) would; optimistic when the strategy needs
    # that very close to decide. "next_close": at the close of bar i+1, one bar late, like an order left for the next
    # session's ATC. Paper always fills at the next open or the live price.
    fill_at: str = "open"

    def as_dict(self) -> dict:
        return {
            "fill_at": self.fill_at,
            "fixed_quantity": self.fixed_quantity,
            "initial_capital": self.initial_capital,
            "size_pct": self.size_pct,
            "leverage": self.leverage,
            "fee": self.fee,
            "slippage": self.slippage,
            "contract": self.contract.as_dict() if self.contract else None,
            "exits": self.exits.as_dict() if self.exits else None,
            "vn_cash": self.vn_cash.as_dict() if self.vn_cash else None,
            "funding_rate": self.funding_rate,
            "borrow_rate": self.borrow_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BacktestConfig":
        values = dict(data)
        contract = values.pop("contract", None)
        exits = values.pop("exits", None)
        vn_cash = values.pop("vn_cash", None)
        return cls(**values, contract=ContractConfig.from_dict(contract) if contract else None,
                   exits=ExitConfig.from_dict(exits), vn_cash=VnCashRules.from_dict(vn_cash))


@dataclass
class Trade:
    side: str            # "long" | "short"
    entry_index: int
    exit_index: int
    entry_time: int      # epoch seconds
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float    # on the margin committed
    bars_held: int
    # "signal" | "liquidation" | "end_of_data" | "stop_loss" | "trailing_stop"
    # | "take_profit" | "take_profit_partial"
    exit_reason: str
    # Excursions, in percent of the margin committed — the same base as
    # ``return_pct``, so the three numbers are directly comparable.
    #
    # MFE is how far the trade went in your favour before it closed, MAE how
    # far against. They are what turns "this stop is too tight" from an opinion
    # into a measurement: if winning trades routinely sit 4% underwater first,
    # a 3% stop cuts the winners, not the losers.
    #
    # The entry bar counts. The fill happens at its open, so the rest of that
    # bar's range genuinely occurs while the position is held; the only thing
    # the engine cannot know is the order of that bar's high and low, and order
    # does not change a maximum. Excluding it would understate MAE — and an
    # understated MAE is the dangerous direction, because it makes a stop look
    # safer than it is. This also matches the liquidation check, which already
    # tests the entry bar's own extreme.
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    # The same excursions as a price move in points, signed like ``points`` (2026-10-30: a futures report reads in
    # points, and a % of the margin beside a profit in points could not be compared). Fees are not in them.
    mfe_points: float = 0.0
    mae_points: float = 0.0
    # Account equity immediately before this trade was opened, and immediately
    # after it closed.
    #
    # These exist so a resampler can work in the same units the engine does.
    # `return_pct` is measured on the margin committed and excludes the entry
    # fee, which is charged the moment the position opens — so compounding it
    # does not reproduce the equity curve (measured 4-7% out). The fractional
    # change the account actually saw is `equity_after / equity_before - 1`,
    # and nothing else in the record carries it.
    equity_before: float = 0.0
    equity_after: float = 0.0
    # The price move in points, signed by side. Both models carry it; the
    # contract model also records the whole contracts held and the multiplier
    # (dong per point) that turned points into money.
    points: float = 0.0
    contracts: int | None = None
    multiplier: float | None = None
    # Sale tax paid on this exit (VN cash rules only), already inside ``pnl``.
    tax: float = 0.0
    # Funding paid (negative) or received (positive) since the previous record, inside ``pnl``.
    funding: float = 0.0
    # The entry fee this record carries (the whole fee on a full close, its share on a partial one) and the profit
    # after EVERY cost: ``pnl - entry_fee``. ``pnl`` keeps its old meaning (price move, exit fee, tax, funding), because
    # the entry fee leaves the account when the position opens, not when it closes; but win/loss counts, profit factor
    # and the averages must be taken after it, or a trade that loses money on the account counts as a win (audit R2-01:
    # a 0.15% move with a 0.1% fee each way was a "win" while the account fell). ``None`` on records from before this.
    entry_fee: float = 0.0
    net_pnl: float | None = None
    # ``net_pnl`` on the margin committed, the figure the trade table shows as a percentage: ``return_pct`` leaves the
    # entry fee out, so a trade that lost money on the account could still read as a gain (audit R2-01 follow-up).
    net_return_pct: float | None = None
    # Index futures sized by % of capital: where the broker would have
    # force-sold this position ({"time", "price"}). The backtest does not
    # close it there (Nam, 2026-10-14); the mark says this sizing is too big.
    liquidation_warning: dict | None = None
    # The fee paid on this exit, already inside ``pnl`` (the entry fee is ``entry_fee``).
    exit_fee: float = 0.0
    # VN cash rules only: {"shares", "sellable_time", "waited_settlement", "floor_bars"} - the shares this record
    # closed, when they could first be sold (T+2.5, epoch s, None past the data), whether the strategy's sale had to
    # wait for them, and how many bars locked at the floor refused it. None without the rules.
    vn: dict | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BacktestResult:
    equity: np.ndarray = field(default_factory=lambda: np.array([]))
    times: list[int] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    position: np.ndarray = field(default_factory=lambda: np.array([]))
    config: BacktestConfig = field(default_factory=BacktestConfig)
    liquidated: bool = False
    ruined: bool = False     # equity hit zero; trading stopped
    # What the VN cash rules changed, when they applied; None otherwise.
    vn: dict | None = None
    # Index futures only: how close the account came to a force sale, measured, not assumed (Nam, 2026-10-17).
    # lowest_ratio: the lowest of equity / contract value, at each bar's worst price while a position was open.
    # capital_needed: the least starting capital that would never have met the force-sell rate nor been refused
    # an entry, for a fixed contract count on a path no force sale or refusal changed; None otherwise.
    margin: dict | None = None


def _vn_calendar(open_ms: np.ndarray, high: np.ndarray, low: np.ndarray,
                 close: np.ndarray, vn: VnCashRules):
    """Per-bar facts the VN rules need, read from the bars themselves.

    * ``session``: the trading session each bar belongs to, counted over the
      dates that have bars (holidays have none, so they are skipped). A day
      missing from the data would count one session short: the settlement
      then looks a session later than it was.
    * ``after_settle``: the bar opens at or after 13:00 Vietnam time, when
      shares bought two sessions earlier arrive. A daily bar is stamped
      before that, so on daily bars the first sale is session T+3.
    * ``locked_down`` / ``locked_up``: the bar traded at one price, at least
      ``lock_move`` below (above) the previous session's close. HOSE's band is
      7%, HNX 10%, UPCOM 15%, and the exchange is not in the data: a one-price
      bar 6.9% down is taken as locked everywhere. OHLC cannot show an empty
      bid side, so a floor that still had buyers is read as locked too.
    """
    vn_time = pd.to_datetime(open_ms, unit="ms") + pd.Timedelta(hours=7)
    dates = vn_time.normalize()
    session = pd.factorize(dates, sort=True)[0]
    after_settle = np.asarray(vn_time.hour >= vn.settle_hour_vn)

    last_close = pd.Series(close).groupby(session).last()
    reference = pd.Series(session).map(last_close.shift(1)).to_numpy(dtype="float64")
    one_price = np.abs(high - low) <= 1e-9 * np.maximum(np.abs(close), 1.0)
    with np.errstate(invalid="ignore"):
        locked_down = one_price & (low <= reference * (1.0 - vn.lock_move))
        locked_up = one_price & (high >= reference * (1.0 + vn.lock_move))
    return session, after_settle, np.nan_to_num(locked_down, nan=0).astype(bool), \
        np.nan_to_num(locked_up, nan=0).astype(bool)


def _closes_after_settle(open_ms: np.ndarray, vn: VnCashRules) -> np.ndarray:
    """Per bar: its close comes at or after the settlement hour (13:00 Vietnam time). Used when signals fill at the
    close. A bar ends one bar length after it opens (the median step between opens), and no later than 15:00 of its
    day, so a daily bar ends at the session's close."""
    vn_open = pd.to_datetime(open_ms, unit="ms") + pd.Timedelta(hours=7)
    step = pd.Timedelta(milliseconds=int(np.median(np.diff(open_ms)))) if len(open_ms) > 1 else pd.Timedelta(minutes=1)
    ends = np.minimum((vn_open + step).to_numpy(), (vn_open.normalize() + pd.Timedelta(hours=15)).to_numpy())
    return np.asarray(pd.DatetimeIndex(ends).hour >= vn.settle_hour_vn)


def _vn_exchange(vn: VnCashRules, df: pd.DataFrame):
    """The rules with the exchange's band and price step filled in."""
    from dataclasses import replace

    from backend.data import exchange

    if vn.exchange_info:
        return vn, vn.exchange_info
    if vn.exchange in exchange.BANDS:
        info = {"code": vn.exchange, "band": exchange.BANDS[vn.exchange],
                "tick": exchange.TICKS[vn.exchange], "evidence": {"forced": True}}
    else:
        info = exchange.infer(df)
    # ETFs and warrants keep their own 10 đ step whatever the exchange.
    tick = vn.tick if vn.tick == "flat" else info["tick"]
    return replace(vn, lock_move=info["band"] - 0.001, tick=tick, exchange_info=info), info


def run_backtest(
    df: pd.DataFrame,
    signal: np.ndarray,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate ``signal`` over ``df`` and return equity, positions and trades.

    ``signal`` holds -1 / 0 / 1 per candle, aligned to ``df``. ``df`` needs
    ``open_time, open, high, low, close``.
    """
    config = config or BacktestConfig()
    n = len(df)
    if n == 0:
        return BacktestResult(config=config)

    # Sizing, fees, P&L and the forced-close price come from the position
    # model, shared with the paper engine (backend/strategy/position_model.py).
    model = model_for(config)

    open_ = df["open"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    times = (df["open_time"].to_numpy(dtype="int64") // 1000).tolist()

    # A signal computed on bar i is actionable only from bar i+1's open; at the close (fill_at "close") it acts on bar
    # i itself, at the end of the bar, after that bar's stops and liquidation (step 2b of the loop); "next_close" acts
    # the same way one bar later, at the close of bar i+1.
    at_close = config.fill_at in ("close", "next_close")
    if config.fill_at == "close":
        target = np.asarray(signal, dtype="int8").copy()
    else:
        target = np.zeros(n, dtype="int8")
        target[1:] = signal[:-1]

    vn = config.vn_cash
    vn_stats = None
    exchange_info = None
    if vn is not None and vn.exchange != "none":
        vn, exchange_info = _vn_exchange(vn, df)
    if vn is not None:
        # No short selling on the cash market: a short signal is flat. Counted,
        # because a strategy built to go short loses half of itself here.
        vn_stats = {
            "shorts_dropped": int(np.sum((target < 0) & (np.concatenate([[0], target[:-1]]) >= 0))),
            "exits_deferred_settlement": 0,   # trades whose exit waited for T+2.5
            "sells_blocked_floor": 0,         # bars a sale could not fill: locked at the floor
            "buys_blocked_ceiling": 0,        # bars a purchase could not fill: locked at the ceiling
            "entries_below_lot": 0,           # entries skipped: not one whole lot affordable
            "tax_paid": 0.0,
            # Which exchange's band and price step were used, and why.
            "exchange": exchange_info,
        }
        target = np.where(target < 0, 0, target).astype("int8")
        session, after_settle, locked_down, locked_up = _vn_calendar(
            df["open_time"].to_numpy(dtype="int64"), high, low, close, vn)
        if at_close:
            # A sale at the close happens when the bar ends, not when it opens: the shares of T arrive at 13:00 of
            # T+2, so the close of the 11:29 bar is too early and the close of a daily bar (the session's end) is not.
            after_settle = _closes_after_settle(df["open_time"].to_numpy(dtype="int64"), vn)
        deferred_trades: set[int] = set()     # entry bars whose exit waited for T+2.5
        floor_bars: set[int] = set()          # bars a sale was refused: locked at the floor
        ceiling_bars: set[int] = set()        # bars a purchase was refused: locked at the ceiling

    equity = config.initial_capital
    margin_low = None      # futures: lowest equity / contract value seen (BacktestResult.margin)
    capital_need = 0.0     # futures: least starting capital that keeps every bar above the force-sell rate
    entries_refused = 0    # futures: signals not filled for want of margin
    equity_curve = np.empty(n, dtype="float64")
    position_curve = np.zeros(n, dtype="int8")

    position = 0          # -1 short, 0 flat, 1 long
    sizing: Sizing | None = None
    entry_price = 0.0
    margin = 0.0
    entry_index = -1
    entry_equity = 0.0    # account equity when the position was opened
    entry_fee_left = 0.0  # the entry fee of the open position not yet booked on a trade record
    best_price = 0.0      # most favourable price seen since entry
    worst_price = 0.0     # least favourable
    trades: list[Trade] = []
    liquidated_ever = False
    ruined = False

    exits = config.exits if config.exits and config.exits.active else None
    atr = wilder_atr(high, low, close) if exits and exits.stop_atr else None
    stop = None            # current stop level, or None
    stop_trails = False    # the stop was last set by the trail
    target_level = None    # current target, or None once a partial took it
    trail_best = 0.0       # best price for the trail, from bars already judged
    blocked = 0            # after a stop/target exit: the direction not to re-enter
    # Funding accrues bar by bar on the linear (crypto) model; the trade record that closes
    # next carries what has accrued since the last record, so pnl includes it.
    funding_on = bool(config.funding_rate or config.borrow_rate) and config.contract is None and vn is None
    funding_hours = float(np.median(np.diff(times))) / 3600.0 if funding_on and n > 1 else 0.0
    funding_since = 0.0

    # Index futures. Force sale follows ContractConfig.force_sale (Nam, 2026-10-17): "account" closes a position when
    # the whole account can no longer carry it, "deal" on the deal's own ratio, as DNSE does without its automatic
    # top-up. From 2026-10-14 to 2026-10-17 neither happened, even with no money left. Under "account", a position
    # sized by % of capital is still marked where the deal on its own would have gone (the yellow star).
    # A continuous series is settled at each expiry it rolls on.
    contract_marks = config.contract is not None
    # Fixed contracts with no force sale (Nam, 2026-10-18, "Setting cấu hình backtesting": "dù còn 10tr vẫn tiếp tục, âm
    # vốn vẫn tiếp tục"): the run is read in points, so it goes on whatever the account holds. No margin check at entry,
    # no stop at zero, and the equity curve may go below zero. Contracts from equity still need equity to be counted.
    keep_going = (contract_marks and config.contract.force_sale == "none" and config.contract.sizing == "fixed")
    floor = (lambda v: v) if keep_going else (lambda v: max(v, 0.0))
    force_rate = config.contract.maintenance_threshold * config.contract.initial_margin_rate if contract_marks else 0.0
    mark_liquidation = contract_marks and config.contract.sizing == "margin" and config.contract.force_sale != "deal"
    warning: dict | None = None     # waiting for the record that closes the position
    warned = False                  # the open position has been marked already
    rolls: set[int] = set()
    if contract_marks and config.contract.series_label and config.contract.expiry != "hold":
        from backend.analysis.vn30f import roll_indices
        rolls = set(roll_indices(df["open_time"].to_numpy(dtype="int64"),
                                 config.contract.series_label).tolist())

    def fill_price(price: float, direction: int) -> float:
        """Worsen a fill by slippage; ``direction`` is +1 when buying."""
        price = model.fill(price, direction)
        return vn_round(price, direction, vn.tick) if vn is not None else price

    def can_sell(index: int, record: bool = False) -> bool:
        """VN rules: the shares have settled (T+2.5) and the bar is not locked
        at the floor. Always true without the rules. ``record`` counts the
        refusal: only a sale the strategy actually asked for is counted, not
        the stop and liquidation checks that run on every bar."""
        if vn is None or position == 0:
            return True
        due = session[entry_index] + vn.settlement_sessions
        settled = session[index] > due or (session[index] == due and after_settle[index])
        if not settled:
            if record:
                deferred_trades.add(entry_index)
            return False
        if locked_down[index]:
            if record:
                floor_bars.add(index)
            return False
        return True

    def sellable_time(index: int) -> int | None:
        """VN rules: the open time of the first bar the shares bought at ``index`` can be sold on (T+2.5)."""
        due = session[index] + vn.settlement_sessions
        ok = (session[index + 1:] > due) | ((session[index + 1:] == due) & after_settle[index + 1:])
        later = np.flatnonzero(ok)
        return int(times[index + 1 + later[0]]) if later.size else None

    def can_sell_settled(index: int) -> bool:
        """VN rules: the shares held have settled by bar ``index`` (T+2.5), whatever the bar's lock."""
        due = session[entry_index] + vn.settlement_sessions
        return session[index] > due or (session[index] == due and after_settle[index])

    def close_position(exit_price: float, index: int, reason: str,
                       part: Sizing | None = None) -> None:
        """Close the position, or only ``part`` of it (a partial target)."""
        nonlocal equity, position, sizing, entry_price, margin, entry_index
        nonlocal best_price, worst_price, entry_equity, funding_since, entry_fee_left, warning
        whole = part is None
        closing = sizing if whole else part
        closing_margin = margin if whole else part.margin
        tax = 0.0
        if vn is not None:
            # A sale fills on the price grid, and the sale value is taxed.
            exit_price = vn_round(exit_price, -1, vn.tick)
            tax = vn.sell_tax * abs(closing.quantity) * exit_price
            vn_stats["tax_paid"] += tax
        pnl = model.pnl(closing, entry_price, exit_price) - tax
        equity += pnl
        exit_fee = model.exit_fee(closing, exit_price)
        vn_info = None
        if vn is not None:
            vn_info = {"shares": abs(closing.quantity) / vn.price_unit,
                       "sellable_time": sellable_time(entry_index),
                       "waited_settlement": entry_index in deferred_trades,
                       "floor_bars": sum(1 for b in floor_bars if entry_index < b <= index)}
        # The entry fee was taken from the account when the position opened; book it on the record that closes it, in
        # proportion to the size when the position closes in parts.
        if whole or abs(sizing.quantity) <= 0:
            fee_share = entry_fee_left
        else:
            fee_share = entry_fee_left * min(1.0, abs(closing.quantity) / abs(sizing.quantity))
        entry_fee_left -= fee_share

        # The exit fill is a price the position lived through. A signal exit fills at the next open, which lies outside
        # the ranges tracked so far; when that open gapped past every high (or low) seen, the trade realised more than
        # its MFE (or lost more than its MAE): 7.6% and 15.2% of 5,621 trades on gapping test bars (2026-10-30).
        if position > 0:
            best_price = max(best_price, exit_price)
            worst_price = min(worst_price, exit_price)
        elif position < 0:
            best_price = min(best_price, exit_price)
            worst_price = max(worst_price, exit_price)

        # Excursions on the same base as return_pct: unrealised P&L on the
        # committed margin, gross of the exit fee (which is not owed until the
        # trade actually closes).
        if closing_margin > 0:
            mfe = model.unrealized(closing, entry_price, best_price) / closing_margin * 100.0
            mae = model.unrealized(closing, entry_price, worst_price) / closing_margin * 100.0
        else:
            mfe = mae = 0.0

        trades.append(
            Trade(
                side="long" if position > 0 else "short",
                entry_index=entry_index,
                exit_index=index,
                entry_time=times[entry_index],
                exit_time=times[index],
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=abs(closing.quantity),
                pnl=pnl + funding_since,
                return_pct=((pnl + funding_since) / closing_margin * 100.0) if closing_margin > 0 else 0.0,
                bars_held=index - entry_index,
                exit_reason=reason,
                # Clamped: MFE is favourable-or-nothing, MAE adverse-or-nothing,
                # so a trade that only ever moved one way reports 0 for the
                # other rather than a sign-flipped value.

                mfe_pct=max(mfe, 0.0),
                mae_pct=min(mae, 0.0),
                mfe_points=max((best_price - entry_price) * position, 0.0),
                mae_points=min((worst_price - entry_price) * position, 0.0),
                equity_before=entry_equity,
                equity_after=equity,
                points=(exit_price - entry_price) * position,
                contracts=closing.contracts,
                multiplier=model.multiplier,
                tax=tax,
                funding=funding_since,
                entry_fee=fee_share,
                net_pnl=pnl + funding_since - fee_share,
                net_return_pct=((pnl + funding_since - fee_share) / closing_margin * 100.0) if closing_margin > 0 else 0.0,
                liquidation_warning=warning,
                exit_fee=exit_fee,
                vn=vn_info,
            )
        )
        funding_since = 0.0
        warning = None

        if not whole:
            # The rest stays open. The next trade in the chain starts from the
            # equity this one ended on, so equity_after / equity_before still
            # compounds to the account's path.
            margin -= part.margin
            sizing = model.restore(sizing.quantity - part.quantity, margin, entry_price)
            entry_equity = equity
            return

        position = 0
        sizing = None
        entry_price = 0.0
        margin = 0.0
        entry_index = -1
        best_price = worst_price = 0.0

    def open_position(price: float, index: int, direction: int) -> None:
        nonlocal equity, position, sizing, entry_price, margin, entry_index
        nonlocal best_price, worst_price, entry_equity, entry_fee_left, warned, entries_refused, capital_need
        new = model.size(equity, price, direction)
        if new is None and keep_going:
            # The fixed count opens whatever the account holds (keep_going above); its margin is still recorded.
            new = model.size(math.inf, price, direction)
        if new is None:
            # Not enough margin for one contract (or for the fixed count):
            # the signal is not filled.
            entries_refused += 1
            return
        if config.contract is not None:
            capital_need = max(capital_need, config.initial_capital - equity + new.margin)
        if vn is not None:
            # Whole lots only, rounded down so the order stays affordable.
            shares = math.floor(abs(new.quantity) / vn.price_unit / vn.lot_shares + 1e-9) * vn.lot_shares
            if shares < vn.lot_shares:
                vn_stats["entries_below_lot"] += 1
                return
            quantity = shares * vn.price_unit * direction
            new = model.restore(quantity, abs(quantity) * price / config.leverage, price)
        entry_equity = equity          # before the entry fee is taken
        warned = False
        sizing = new
        margin = new.margin
        entry_price = price
        entry_index = index
        position = direction
        # Start both excursions at the entry price: a trade that closes before
        # any further bar has moved neither way.
        best_price = worst_price = price
        entry_fee_left = model.entry_fee(new, price)
        equity -= entry_fee_left                # entry fee, paid immediately
        if exits:
            arm_exits(price, index, direction)

    def arm_exits(price: float, index: int, direction: int) -> None:
        """Place the stop and target for a fresh position."""
        nonlocal stop, stop_trails, target_level, trail_best
        fixed = []
        if exits.stop_pct:
            fixed.append(price * (1.0 - direction * exits.stop_pct / 100.0))
        # ATR of the bars before this one: the entry bar has only opened. Filled at the close, the entry bar is over.
        known = index if at_close else index - 1
        if exits.stop_atr and atr is not None and known >= 0 and np.isfinite(atr[known]):
            fixed.append(price - direction * exits.stop_atr * atr[known])
        tightest = max if direction > 0 else min
        stop = tightest(fixed) if fixed else None
        stop_trails = False
        if exits.trail_pct:
            # Armed at the fill, like the paper engine's.
            level = price * (1.0 - direction * exits.trail_pct / 100.0)
            if stop is None or tightest(level, stop) == level:
                stop, stop_trails = level, True
        target_level = (price * (1.0 + direction * exits.target_pct / 100.0)
                        if exits.target_pct else None)
        trail_best = price

    def partial_sizing(fraction: float) -> Sizing | None:
        """The share a partial target closes, or None when it rounds to none
        or to all of the position (whole contracts, or whole lots under the
        VN rules)."""
        if vn is not None:
            held = abs(sizing.quantity) / vn.price_unit
            part = math.floor(held * fraction / vn.lot_shares + 1e-9) * vn.lot_shares
            if part < vn.lot_shares or part >= held - 1e-9:
                return None
            share = part / held
            return model.restore(part * vn.price_unit * position, margin * share, entry_price)
        if sizing.contracts is not None:
            count = int(sizing.contracts * fraction)
            if count < 1 or count >= sizing.contracts:
                return None
            share = count / sizing.contracts
            qty = float(count) * position
        else:
            share, qty = fraction, sizing.quantity * fraction
        return model.restore(qty, margin * share, entry_price)

    def judge_exits(i: int) -> None:
        """Stops and targets against bar ``i``, with the levels as they stood
        before it (a level moved inside this bar waits for the next)."""
        nonlocal stop, stop_trails, target_level, trail_best, blocked
        direction = position
        liq = (model.force_sale_price(sizing, entry_price, direction, equity) if contract_marks
               else model.liquidation_price(entry_price, direction))
        # A stop past the forced-close price is never reached before it.
        stop_live = stop is not None and (
            liq is None or (stop > liq if direction > 0 else stop < liq))
        reason = "trailing_stop" if stop_trails else "stop_loss"
        if stop_live and i > entry_index:
            gapped = open_[i] <= stop if direction > 0 else open_[i] >= stop
            if gapped:
                close_position(fill_price(open_[i], -direction), i, reason)
                blocked = direction
                return
        # A target the bar opened past fills at that open, the first price there was, and before anything later in the
        # bar (Nam, 2026-10-29: "sửa hết"). It filled at its own level until then, a price the bar never traded: a VN
        # long held through T+2.5 while the price ran from 24.55 to 31 took its target at 24.55.
        target_gapped = target_level is not None and i > entry_index and (
            open_[i] >= target_level if direction > 0 else open_[i] <= target_level)
        if not target_gapped and stop_live and (low[i] <= stop if direction > 0 else high[i] >= stop):
            # A stop turns into a market order once touched, so it fills at the
            # level worsened by slippage (Nam, 2026-10-07). A target is a resting
            # limit order and fills at its own level, or at a better open.
            close_position(fill_price(stop, -direction), i, reason)
            blocked = direction
            return
        if target_gapped or (target_level is not None and (
                high[i] >= target_level if direction > 0 else low[i] <= target_level)):
            filled = float(open_[i]) if target_gapped else target_level
            part = partial_sizing(exits.tp_fraction) if exits.tp_fraction < 1.0 else None
            if part is None:
                close_position(filled, i, "take_profit")
                blocked = direction
                return
            close_position(filled, i, "take_profit_partial", part)
            target_level = None
            if exits.tp_breakeven and (
                    stop is None or (entry_price > stop if direction > 0 else entry_price < stop)):
                stop, stop_trails = entry_price, False
        ratchet_trail(i)

    def ratchet_trail(i: int) -> None:
        """Ratchet the trail from this bar's favourable extreme, for the next bar.

        Also runs on bars where the shares cannot be sold yet (T+2.5, or a
        floor-locked bar): the stop cannot fire then, but the price still moved,
        and a trail that ignored those bars would sit far below the real
        high-water mark when the shares finally become sellable (Nam, 2026-10-07).
        """
        nonlocal stop, stop_trails, trail_best
        if not exits.trail_pct:
            return
        direction = position
        p = exits.trail_pct / 100.0
        if direction > 0:
            trail_best = max(trail_best, high[i])
            level = trail_best * (1.0 - p)
            if stop is None or level > stop:
                stop, stop_trails = level, True
        else:
            trail_best = min(trail_best, low[i])
            level = trail_best * (1.0 + p)
            if stop is None or level < stop:
                stop, stop_trails = level, True

    def act(i: int, wanted: int, price: float, may_open: bool = True) -> None:
        """Move the position to ``wanted`` at ``price`` (worsened by slippage). After a stop or target, the same
        direction waits for the signal to change before it is taken again."""
        nonlocal blocked
        if blocked:
            if wanted == blocked:
                wanted = position
            else:
                blocked = 0
        if wanted != position:
            # Under the VN rules a sale waits for T+2.5 and for a bar not
            # locked at the floor, and a purchase for one not locked at the
            # ceiling; the order is simply tried again on the next bar.
            flat = True
            if position != 0:
                if can_sell(i, record=True):
                    close_position(fill_price(price, -position), i, "signal")
                else:
                    flat = False
            if flat and wanted != 0 and may_open and (equity > 0 or keep_going):
                if vn is not None and locked_up[i]:
                    ceiling_bars.add(i)
                else:
                    open_position(fill_price(price, wanted), i, wanted)

    def charge_funding(i: int) -> None:
        """Funding for the bar the position was held through."""
        nonlocal equity, funding_since
        if funding_on and position != 0 and sizing is not None:
            value = abs(sizing.quantity) * close[i]
            cost = position * value * config.funding_rate * funding_hours / 8.0
            if position < 0:
                cost += value * config.borrow_rate * funding_hours / (24.0 * 365.0)
            equity -= cost
            funding_since -= cost

    report_every = progress.every(n)
    for i in range(n):
        if i % report_every == 0:
            progress.report(i, n, "simulate")
        if not ruined and i in rolls and position != 0:
            # 0. The contract the series held until the last bar expired there:
            #    the position is settled at that bar's close (no slippage, the
            #    closing fee paid). A signal that still wants it reopens below,
            #    at this bar's open, on the new contract, paying its fees again.
            settled = position
            close_position(close[i - 1], i - 1, "expiry")
            # "close" (ContractConfig.expiry): no new position until the signal changes, as after a stop.
            blocked = settled if config.contract.expiry == "close" else 0
            equity_curve[i - 1] = floor(equity)
            if at_close:
                # Filled at the close, the curve holds each bar's state at its end, as a signal's exit at a close does:
                # the position settled at this close is gone by then (found by the scan of 2026-10-29).
                position_curve[i - 1] = 0
            if equity <= 0 and not keep_going:
                equity = 0.0
                ruined = True

        if not ruined:
            if not at_close:
                # 1. Act on the previous bar's signal, at this bar's open.
                act(i, int(target[i]), open_[i])
            elif i in rolls and i >= 1:
                # Filled at the close, nothing opens at the close of an expiring contract (step 3); the new contract
                # is taken at its first open when the last signal still wants a position.
                act(i, int(target[i - 1]), open_[i])

            # 1b. Track how far this bar took the open position either way.
            #     After the fill, so the entry bar is measured from the price
            #     actually paid; before liquidation, so the wick that ends a
            #     trade is still counted in its MAE.
            if position != 0:
                if position > 0:
                    best_price = max(best_price, high[i])
                    worst_price = min(worst_price, low[i])
                else:
                    best_price = min(best_price, low[i])
                    worst_price = max(worst_price, high[i])

            # 1c. Stops and targets, before liquidation: a stop between the
            #     entry and the forced-close price is passed first.
            if exits and position != 0:
                if can_sell(i):
                    judge_exits(i)
                else:
                    # A stop or target this bar reached, refused by a floor lock (the shares had settled): counted
                    # with the signal sales it refused, so the trade says why it sold below its stop (2026-10-15).
                    if vn is not None and locked_down[i] and can_sell_settled(i) and (
                            (stop is not None and low[i] <= stop) or
                            (target_level is not None and high[i] >= target_level)):
                        floor_bars.add(i)
                    ratchet_trail(i)

            # 2. Liquidation: does this bar's adverse extreme reach the price at
            #    which the position is force-closed? None means no such price
            #    (the linear model at leverage 1).
            #    A bar that opens past it never traded there: the position is
            #    closed at that open, worsened by slippage, and the loss is the
            #    one the gap made, not the one the level promised (Nam,
            #    2026-09-26). The entry bar is exempt: its open is the fill.
            if position != 0 and can_sell(i):
                if mark_liquidation and not warned:
                    # Where the deal on its own would have been force-sold (DNSE without its top-up): a mark only.
                    deal = model.liquidation_price(entry_price, position)
                    deal_gap = i > entry_index and (open_[i] <= deal if position > 0 else open_[i] >= deal)
                    if deal_gap or (low[i] <= deal if position > 0 else high[i] >= deal):
                        warned = True
                        warning = {"time": times[i], "price": float(open_[i] if deal_gap else deal)}
                liq_price = (model.force_sale_price(sizing, entry_price, position, equity) if contract_marks
                             else model.liquidation_price(entry_price, position))
                if liq_price is not None:
                    gapped = i > entry_index and (
                        open_[i] <= liq_price if position > 0 else open_[i] >= liq_price)
                    hit = low[i] <= liq_price if position > 0 else high[i] >= liq_price
                    if contract_marks:
                        if gapped or hit:
                            # A futures force sale books at the level, or at the open a gap jumped to, with no
                            # slippage, as in Paper (Nam, 2026-10-14).
                            close_position(float(open_[i]) if gapped else liq_price, i, "liquidation")
                            liquidated_ever = True
                    elif gapped:
                        close_position(fill_price(open_[i], -position), i, "liquidation")
                        liquidated_ever = True
                    elif hit:
                        close_position(liq_price, i, "liquidation")
                        liquidated_ever = True

            # 2b. Filled at the close: this bar's own signal, now that the bar is over, at its close. After the bar's
            #     stops and liquidation, so a position opened here meets none of this bar's range (its high, low and
            #     open came before the fill) and its funding starts with the next bar. Nothing opens on the last bar
            #     (no bar left to hold it) nor at the close of an expiring contract (reopened at the next open above).
            entered_at_close = False
            if at_close:
                charge_funding(i)
                held_before = (position, entry_index)
                act(i, int(target[i]), close[i], may_open=not (i == n - 1 or (i + 1) in rolls))
                entered_at_close = position != 0 and (position, entry_index) != held_before

            if equity <= 0 and not keep_going:
                equity = 0.0
                ruined = True
        else:
            entered_at_close = False

        # 3. Funding for the bar the position was held through (filled at the close: charged in 2b), then mark to
        #    market on its close.
        if not at_close:
            charge_funding(i)
        unrealized = model.unrealized(sizing, entry_price, close[i]) if position != 0 else 0.0
        if contract_marks and position != 0 and not entered_at_close:
            worst = low[i] if position > 0 else high[i]
            value = abs(sizing.contracts or 0) * model.multiplier * worst
            if value > 0:
                at_worst = equity + model.unrealized(sizing, entry_price, worst)
                margin_low = at_worst / value if margin_low is None else min(margin_low, at_worst / value)
                capital_need = max(capital_need, config.initial_capital - at_worst + force_rate * value)
        equity_curve[i] = floor(equity + unrealized)
        position_curve[i] = position

    # Close any position still open, so the trade list is complete.
    if position != 0:
        close_position(fill_price(close[n - 1], -position), n - 1, "end_of_data")
        equity_curve[n - 1] = floor(equity)

    if vn_stats is not None:
        vn_stats["exits_deferred_settlement"] = len(deferred_trades)
        vn_stats["sells_blocked_floor"] = len(floor_bars)
        vn_stats["buys_blocked_ceiling"] = len(ceiling_bars)

    margin_facts = None
    if contract_marks:
        path_kept = not liquidated_ever and not ruined and entries_refused == 0 and config.contract.sizing == "fixed"
        margin_facts = {
            "lowest_ratio": float(margin_low) if margin_low is not None else None,
            "force_sale_rate": force_rate,
            "capital_needed": float(capital_need) if path_kept and trades else None,
            "entries_refused": entries_refused,
            "liquidations": sum(1 for t in trades if t.exit_reason == "liquidation"),
        }
    return BacktestResult(
        equity=equity_curve,
        times=times,
        trades=trades,
        position=position_curve,
        config=config,
        liquidated=liquidated_ever,
        ruined=ruined,
        vn=vn_stats,
        margin=margin_facts,
    )
