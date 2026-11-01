"""How a position is sized, charged and marked.

Two models behind one interface, used by both the backtest engine and the paper
engine, so the execution rules (CLAUDE.md §3.1) are written in one place.

* LinearModel: crypto, equities, indices, commodities. Margin = equity x
  size_pct, notional = margin x leverage, quantity = notional / price, fees a
  fraction of notional. The same arithmetic the engines used before this
  module; tests/test_position_model.py holds it to a recorded run.
* ContractModel: Vietnamese index futures (market_vn class ``futures_vn``).
  Capital in dong, whole contracts, P&L = points x multiplier x contracts,
  margin by rate, forced close by a margin threshold.

Design: docs/superpowers/specs/2026-09-15-contract-position-model-design.md
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

EXECUTION_LINEAR = "linear"
EXECUTION_CONTRACT = "contract"
EXECUTION_CONTRACT_OFF = "contract_model_off"


@dataclass
class ContractConfig:
    initial_margin_rate: float          # (0, 1]
    maintenance_threshold: float        # [0, 1): remaining margin share that forces a close
    sizing: str = "margin"              # "margin" | "fixed"
    contracts: int = 1                  # used when sizing == "fixed"
    multiplier: float = 100_000.0       # dong per index point
    fee_mode: str = "per_contract"      # "per_contract" | "notional"
    fee_per_contract: float = 0.0       # dong per contract, per side
    fee_rate: float = 0.0               # fraction of notional, per side
    slippage_points: float = 0.0        # adverse, per fill
    # The most contracts one position may hold; None means no cap. Sizing by margin compounds: a 100 million VND
    # account that doubles a hundred times ends up asking for hundreds of thousands of contracts, which the market
    # cannot fill at one price. A backtest that did so showed 814 trillion million VND of profit (Nam, 2026-10-13).
    max_contracts: int | None = 1000
    # Which continuous series this is ("F1M", "F2M", "F1Q", "F2Q"), so the
    # engines know when it switches contract and close a position held into
    # the switch at expiry; None for one dated contract (series_label_for).
    series_label: str | None = None
    # When the broker force-sells (Nam, 2026-10-17, from DNSE's own rules: hdsd.dnse.com.vn, FAQ phai sinh). DNSE
    # judges each deal on its own ratio, and can move free cash into the deal automatically ("tang ty le deal tu
    # dong"). "account": that top-up is on, so a position goes only when the whole account can no longer carry it,
    # which every broker does at the least. "deal": no top-up, the deal goes on its own ratio (a long after a 1.37%
    # fall, a short after a 0.96% rise at 18.48% / 17.35%). From 2026-10-14 to 2026-10-17 the backtest force-sold
    # neither way, even an account with no money left.
    # "none" (the default from 2026-10-17, Nam: "backtest y chang như amibroker"): like AmiBroker, margin is checked
    # when a position opens and never while it is held. The two DNSE rules are options.
    force_sale: str = "none"
    # What a backtest does with a position held into a continuous series' expiry (Nam, 2026-10-29: "cắt lệnh chiến lược
    # khi đáo hạn, hoặc không cắt, sau khi cắt lệnh thì có vào lại hay không"). "reopen", the default: settled at that
    # close and taken again at the new contract's open when the signal still wants it, paying both fees again, as a
    # trader has to. "close": settled, then flat until the strategy's signal changes, as after a stop. "hold": held
    # through, the series read as one instrument, as AmiBroker reads it. Paper always settles and reopens: a real
    # position cannot outlive its contract.
    expiry: str = "reopen"
    # Personal income tax on a futures trade, each side (Nam, 2026-10-30, the Scenarios screen): the rate on half the
    # initial margin of the contracts traded, i.e. tax_rate × price × multiplier × contracts × initial margin ÷ 2
    # (0.1% in the VN rules found then; Thông tư 87/2026/TT-BTC not read in the original). 0 by default: a backtest
    # charges it only when asked, so earlier results do not move.
    tax_rate: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ContractConfig":
        return cls(**{"tax_rate": 0.0, **data})


@dataclass(frozen=True)
class Sizing:
    quantity: float           # signed: base units (linear) or contracts (contract)
    margin: float             # committed to the position
    notional: float           # at entry
    contracts: int | None = None


class LinearModel:
    kind = EXECUTION_LINEAR
    multiplier = None

    def __init__(self, config) -> None:
        self.config = config

    def fill(self, price: float, direction: int) -> float:
        return price * (1.0 + self.config.slippage * direction)

    def size(self, equity: float, price: float, direction: int,
             size_pct: float | None = None) -> Sizing | None:
        fixed = getattr(self.config, "fixed_quantity", None)
        if fixed and size_pct is None:
            # The same quantity every time; refused, like a contract, when the account cannot put up its margin.
            notional = fixed * price
            margin = notional / self.config.leverage
            if margin > equity:
                return None
            return Sizing(quantity=fixed * direction, margin=margin, notional=notional)
        margin = equity * (self.config.size_pct if size_pct is None else size_pct)
        notional = margin * self.config.leverage
        return Sizing(quantity=notional / price * direction, margin=margin, notional=notional)

    def restore(self, quantity: float, margin: float, entry_price: float) -> Sizing:
        return Sizing(quantity=quantity, margin=margin, notional=margin * self.config.leverage)

    def entry_fee(self, sizing: Sizing, price: float) -> float:
        return sizing.notional * self.config.fee

    def exit_fee(self, sizing: Sizing, exit_price: float) -> float:
        return abs(sizing.quantity) * exit_price * self.config.fee

    def unrealized(self, sizing: Sizing, entry_price: float, mark: float) -> float:
        return sizing.quantity * (mark - entry_price)

    def pnl(self, sizing: Sizing, entry_price: float, exit_price: float) -> float:
        return sizing.quantity * (exit_price - entry_price) - self.exit_fee(sizing, exit_price)

    def liquidation_price(self, entry_price: float, direction: int) -> float | None:
        if self.config.leverage <= 1.0:
            return None
        return entry_price * (1.0 - direction / self.config.leverage)


class ContractModel:
    kind = EXECUTION_CONTRACT

    def __init__(self, config) -> None:
        self.config = config
        self.contract: ContractConfig = config.contract
        self.multiplier = self.contract.multiplier

    def fill(self, price: float, direction: int) -> float:
        return price + self.contract.slippage_points * direction

    def margin_per_contract(self, price: float) -> float:
        return price * self.multiplier * self.contract.initial_margin_rate

    def size(self, equity: float, price: float, direction: int,
             size_pct: float | None = None) -> Sizing | None:
        per = self.margin_per_contract(price)
        if per <= 0:
            return None
        if self.contract.sizing == "fixed":
            contracts = int(self.contract.contracts)
        else:
            share = self.config.size_pct if size_pct is None else size_pct
            contracts = math.floor(equity * share / per)
        cap = self.contract.max_contracts
        if cap is not None and contracts > cap:
            contracts = int(cap)
        # Not enough margin for one contract, or for the fixed count: no fill.
        if contracts < 1 or contracts * per > equity:
            return None
        return Sizing(quantity=float(contracts * direction), margin=contracts * per,
                      notional=contracts * price * self.multiplier, contracts=contracts)

    def restore(self, quantity: float, margin: float, entry_price: float) -> Sizing:
        contracts = int(round(abs(quantity)))
        return Sizing(quantity=quantity, margin=margin,
                      notional=contracts * entry_price * self.multiplier, contracts=contracts)

    def _fee(self, contracts: int, price: float) -> float:
        tax = contracts * price * self.multiplier * self.contract.initial_margin_rate * self.contract.tax_rate / 2.0
        if self.contract.fee_mode == "notional":
            return contracts * price * self.multiplier * self.contract.fee_rate + tax
        return contracts * self.contract.fee_per_contract + tax

    def entry_fee(self, sizing: Sizing, price: float) -> float:
        return self._fee(sizing.contracts, price)

    def exit_fee(self, sizing: Sizing, exit_price: float) -> float:
        return self._fee(sizing.contracts, exit_price)

    def unrealized(self, sizing: Sizing, entry_price: float, mark: float) -> float:
        return sizing.quantity * (mark - entry_price) * self.multiplier

    def pnl(self, sizing: Sizing, entry_price: float, exit_price: float) -> float:
        return self.unrealized(sizing, entry_price, exit_price) - self.exit_fee(sizing, exit_price)

    def liquidation_price(self, entry_price: float, direction: int) -> float | None:
        """The price at which the broker force-sells the position.

        DNSE judges each deal on its net assets over its value at the *current*
        price: (margin + unrealized) / (price x multiplier) <= force-sell rate,
        with margin = initial rate x entry value and force-sell rate =
        threshold x initial rate. Solved for the price:

            long   P0 (1 - IM) / (1 - FS)        short   P0 (1 + IM) / (1 + FS)

        At 18.48% / 17.35% a long goes after a 1.37% fall and a short after a
        0.96% rise. Until 2026-10-14 the value at entry was the denominator,
        which put both at 1.13%. The estimated closing fee DNSE also subtracts
        is left out: a few thousand dong a contract against tens of millions.
        """
        c = self.contract
        im = c.initial_margin_rate
        fs = c.maintenance_threshold * im
        return entry_price * (1.0 - direction * im) / (1.0 - direction * fs)

    def force_sale_price(self, sizing: Sizing, entry_price: float, direction: int, account: float) -> float | None:
        """The price at which the broker force-sells, under the chosen rule (ContractConfig.force_sale).

        "deal": liquidation_price, the deal on its own. "account": the cash the account holds after the entry fee
        (``account``) backs the deal, so the sale comes when account + unrealized falls to the force-sell rate of the
        value at the current price:

            account + d x q x m x (P - P0) = fs x q x m x P   =>   P = (d x P0 - account / (q x m)) / (d - fs)

        A long whose account could lose the whole contract value and still cover the rate has no such price (None).
        """
        if self.contract.force_sale == "none":
            return None
        if self.contract.force_sale == "deal":
            return self.liquidation_price(entry_price, direction)
        c = self.contract
        fs = c.maintenance_threshold * c.initial_margin_rate
        exposure = abs(sizing.contracts or 0) * self.multiplier
        if exposure <= 0:
            return None
        price = (direction * entry_price - account / exposure) / (direction - fs)
        return price if price > 0 else None


def model_for(config) -> LinearModel | ContractModel:
    return ContractModel(config) if getattr(config, "contract", None) else LinearModel(config)


def is_index_future(symbol: str | None) -> bool:
    """True for a Vietnamese index future such as ``VN:VN30F1M``."""
    name = (symbol or "").strip()
    if not name.upper().startswith("VN:"):
        return False
    # Imported here: the data module is only needed for this question.
    from backend.data.market_vn import classify
    return classify(name[3:]) == "futures_vn"


def additive_sizing(contract) -> bool:
    """True when each trade's profit is a fixed amount of money rather than a share of the equity at the time.

    One fixed number of contracts earns the same money per point whatever the account holds, so profits add up. Sizing
    from equity (and every linear run) compounds. A resampler has to know which: compounding the percentage change of
    a fixed-size trade puts a late trade's small percentage, earned on a large account, at the start of a path.
    Accepts a ContractConfig, its dict form, or None.
    """
    if contract is None:
        return False
    sizing = contract.get("sizing") if isinstance(contract, dict) else getattr(contract, "sizing", None)
    return sizing == "fixed"


def additive_run(config) -> bool:
    """True when a run's trades add money rather than compound: a fixed number of contracts, or a fixed quantity of
    coins or shares on the linear model (BacktestConfig.fixed_quantity, 2026-10-17). ``config`` is a BacktestConfig or
    its dict form (a result's ``config``). The audit of 2026-10-17 found the resamplers asking only the contract, so a
    fixed quantity of BTC was compounded."""
    if config is None:
        return False
    get = config.get if isinstance(config, dict) else (lambda key, default=None: getattr(config, key, default))
    contract = get("contract")
    if contract is not None:
        return additive_sizing(contract)
    return bool(get("fixed_quantity"))


def contract_terms(config) -> dict | None:
    """The contract terms a run actually used, for the result and the report to state (None for a linear run).

    A futures backtest charges the fee per contract and the slippage in points set in Settings, not the percentages of
    the linear model; a report that printed the percentages said it had charged costs it had not (found 2026-10-13).
    """
    contract = getattr(config, "contract", None)
    if contract is None:
        return None
    return {"sizing": contract.sizing, "contracts": contract.contracts, "max_contracts": contract.max_contracts,
            "fee_mode": contract.fee_mode, "fee_per_contract": contract.fee_per_contract,
            "fee_rate": contract.fee_rate, "slippage_points": contract.slippage_points,
            "multiplier": contract.multiplier, "expiry": contract.expiry, "tax_rate": contract.tax_rate}


_CONTINUOUS_FUTURE_RE = re.compile(r"^VN(?:30|100)(F[12][MQ])$")


def series_label_for(symbol: str | None) -> str | None:
    """The continuous series an index future is ("F1M", "F2M", "F1Q", "F2Q").

    Each switches to another contract on some expiry days (see
    backend/analysis/vn30f.py, ``roll_indices``). A dated contract (41I1…) is
    one contract whose data ends at its own expiry, so it has no label.
    """
    name = (symbol or "").strip().upper()
    match = _CONTINUOUS_FUTURE_RE.match(name[3:]) if name.startswith("VN:") else None
    return match.group(1) if match else None


def execution_model_for(symbol: str | None, config) -> str:
    if not is_index_future(symbol):
        return EXECUTION_LINEAR
    return EXECUTION_CONTRACT if getattr(config, "contract", None) else EXECUTION_CONTRACT_OFF
