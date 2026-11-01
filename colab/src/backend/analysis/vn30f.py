"""Tools for the VN30 index future (Nam, 2026-09-26, feature #7): the basis
against the index, the expiry calendar, the gap paid when the continuous
contract rolls, and the morning and afternoon sessions measured apart.

All figures are in index points unless marked %, and come from the same
one-minute and daily bars as the chart (``api.v_history_1m`` / ``_1d``):
VN30F1M, the continuous front-month series, and VN30, the index.

What each measures and where it breaks (§2.7):

* **Basis** = VN30F1M − VN30 on the same closed minute. Positive: the future
  trades above the index (contango). It shrinks toward 0 as expiry nears,
  because the final settlement price is the index itself. Breaks: the two
  series tick at different moments inside a minute, so one-minute basis
  carries noise of a few points; a minute present in one series and not the
  other is dropped, never filled. The fair basis (index × (rate − dividend
  yield) × time) is not computed: this database has neither the funding rate
  nor the VN30 dividend schedule.
* **Expiry calendar.** The last trading day of a VN30 future is the third
  Thursday of its month, moved to the business day before when that is a
  holiday. Holidays are not in the data, so a date that falls on one is shown
  a day late; the list says "by the rule". Listed months: the current and next
  month, then the next two quarter months (3, 6, 9, 12). The VSD code
  (41I1…) is inferred from the four codes seen in the database
  (41I1G7000 = July 2026 … 41I1GC000 = December 2026): the year letter from
  A = 2020, the month 1–9 then A, B, C.
* **Roll gap.** The continuous series switches contract after expiry, so its
  price jumps by the spread between the two contracts, a move no one traded.
  Measured on daily closes as the basis on the first session after expiry
  minus the basis on expiry day: the index does not roll, so the market's
  own move cancels. Breaks: it keeps one day of genuine basis change; with
  few expiries in the window the average is a handful of numbers.
* **Sessions.** Each day split into the morning (09:00–11:30), the lunch gap,
  the afternoon (13:00–14:30), the closing auction (14:45), and the overnight
  gap from the previous close. Mean move with its standard error and t, the
  share of days up, the mean range and each part's share of volume. Breaks:
  a t of 2 on 120 days is weak evidence, and testing four parts at once makes
  one of them look significant by chance more often than 5%.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Session windows in minutes after 00:00 UTC (CLAUDE.md §3.6, read from the
# data): morning 02:00–04:29, afternoon 06:00–07:29, ATC one bar at 07:45.
AM = (120, 270)
PM = (360, 450)
ATC = (465, 466)
BUCKETS = ((0, 2), (3, 5), (6, 10), (11, 20), (21, 999))
QUARTERS = (3, 6, 9, 12)


# ------------------------------------------------------------- the calendar

def third_thursday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (3 - first.weekday()) % 7          # Thursday is weekday 3
    return first + timedelta(days=offset + 14)


def expiry(year: int, month: int, holidays: frozenset[date] = frozenset()) -> date:
    day = third_thursday(year, month)
    while day in holidays or day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def front_month(on: date) -> tuple[int, int]:
    """The contract VN30F1M is on a given day: this month's until its expiry
    day (inclusive), then next month's."""
    if on <= expiry(on.year, on.month):
        return on.year, on.month
    return _next_month(on.year, on.month)


def vsd_code(year: int, month: int) -> str:
    letter = chr(ord("A") + year - 2020)
    return f"41I1{letter}{'123456789ABC'[month - 1]}000"


def listed(on: date) -> list[dict]:
    """The four listed contracts on a given day, with their expiries."""
    y, m = front_month(on)
    months = [("F1M", y, m)]
    y2, m2 = _next_month(y, m)
    months.append(("F2M", y2, m2))
    qy, qm = y2, m2
    for label in ("F1Q", "F2Q"):
        qy, qm = _next_month(qy, qm)
        while qm not in QUARTERS:
            qy, qm = _next_month(qy, qm)
        months.append((label, qy, qm))
    out = []
    for label, yy, mm in months:
        day = expiry(yy, mm)
        out.append({"label": label, "ticker": f"VN30F{yy % 100:02d}{mm:02d}", "vsd": vsd_code(yy, mm),
                    "expiry": day.isoformat(), "calendar_days": (day - on).days,
                    "business_days": int(np.busday_count(on, day))})
    return out


def business_days_to_expiry(on: date) -> int:
    y, m = front_month(on)
    return int(np.busday_count(on, expiry(y, m)))


def _contract_of(label: str, on: date) -> str:
    return next(row["ticker"] for row in listed(on) if row["label"] == label)


def roll_indices(open_ms, label: str | None) -> np.ndarray:
    """Indices of the first bar after each expiry on which ``label``'s series
    switches contract (VN30F1M every month; VN30F1Q only when its contract
    moves to F2M, and so on).

    The bar before each index is the last one the old contract traded: a
    position still open there is settled at expiry, at that bar's close (the
    final settlement price is the VN30 close, which the front future converges
    to in the closing auction). The calendar is the rule's (third Thursday);
    when that day is a holiday the real expiry is the session before and no
    session lies between them, so the first bar after is the same.
    """
    open_ms = np.asarray(open_ms, dtype="int64")
    if label is None or open_ms.size < 2:
        return np.array([], dtype="int64")
    vn_days = (open_ms + 7 * 3_600_000) // 86_400_000
    epoch = date(1970, 1, 1)
    first = epoch + timedelta(days=int(vn_days.min()))
    last = epoch + timedelta(days=int(vn_days.max()))
    out = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        day = expiry(year, month)
        year, month = _next_month(year, month)
        if not first <= day < last:
            continue
        if _contract_of(label, day) == _contract_of(label, day + timedelta(days=1)):
            continue
        number = (day - epoch).days
        after = int(np.searchsorted(vn_days, number, side="right"))
        if 0 < after < open_ms.size and vn_days[after - 1] <= number:
            out.append(after)
    return np.array(sorted(set(out)), dtype="int64")


# ------------------------------------------------------------------ helpers

def _vn_date(open_ms: pd.Series) -> pd.Series:
    return (pd.to_datetime(open_ms, unit="ms") + pd.Timedelta(hours=7)).dt.date


def _minute_of_day(open_ms: pd.Series) -> pd.Series:
    t = pd.to_datetime(open_ms, unit="ms")
    return t.dt.hour * 60 + t.dt.minute


def _stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "se": None, "t": None, "share_up": None}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else None
    se = std / math.sqrt(n) if std else None
    # §2.6: a t from a spread that is float noise on flat data is not a t.
    scale = float(np.abs(v).mean()) or 1.0
    t = mean / se if se and se > 1e-9 * scale else None
    return {"n": n, "mean": mean, "std": std, "se": se, "t": t, "share_up": float((v > 0).mean())}


# --------------------------------------------------------------------- basis

def basis(fut: pd.DataFrame, index: pd.DataFrame) -> dict:
    """Basis on matched one-minute bars, its distribution, its path by day,
    and its level by business days to expiry."""
    f = fut[["open_time", "close"]].rename(columns={"close": "fut"})
    s = index[["open_time", "close"]].rename(columns={"close": "idx"})
    both = f.merge(s, on="open_time", how="inner").sort_values("open_time").reset_index(drop=True)
    unmatched = {"future_only": int(len(f) - len(both)), "index_only": int(len(s) - len(both))}
    if both.empty:
        return {"measured": False, "matched": 0, "unmatched": unmatched}
    both["basis"] = both["fut"] - both["idx"]
    both["basis_pct"] = both["basis"] / both["idx"] * 100
    both["day"] = _vn_date(both["open_time"])
    last = both.iloc[-1]
    values = both["basis"].to_numpy()
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    level = float(np.abs(both["idx"]).mean())
    days_left = business_days_to_expiry(last["day"])
    calendar_left = (date.fromisoformat(listed(last["day"])[0]["expiry"]) - last["day"]).days

    daily = both.groupby("day").tail(1)
    by_day = [{"day": d.isoformat(), "basis": float(b), "basis_pct": float(p),
               "days_to_expiry": business_days_to_expiry(d)}
              for d, b, p in zip(daily["day"], daily["basis"], daily["basis_pct"])]
    buckets = []
    dte = np.array([business_days_to_expiry(d) for d in both["day"]])
    for lo, hi in BUCKETS:
        mask = (dte >= lo) & (dte <= hi)
        if mask.any():
            part = both["basis"].to_numpy()[mask]
            buckets.append({"from": lo, "to": hi if hi < 999 else None, "minutes": int(mask.sum()),
                            "mean": float(part.mean()), "median": float(np.median(part))})
    # Annualised only a few days out: over 1–2 days a 1-point basis becomes a
    # huge, meaningless yearly rate (§2.6).
    annual = (float(last["basis_pct"]) * 365 / calendar_left) if calendar_left >= 5 else None
    return {
        "measured": True, "matched": int(len(both)), "unmatched": unmatched,
        "latest": {"time": int(last["open_time"]), "future": float(last["fut"]), "index": float(last["idx"]),
                   "basis": float(last["basis"]), "basis_pct": float(last["basis_pct"]),
                   "business_days_to_expiry": days_left, "calendar_days_to_expiry": calendar_left,
                   "annualised_pct": annual},
        "mean": float(values.mean()), "std": std,
        "z": (float(last["basis"]) - float(values.mean())) / std if std and std > 1e-9 * level else None,
        "percentile": float((values <= float(last["basis"])).mean() * 100),
        "by_day": by_day, "by_days_to_expiry": buckets,
    }


# ----------------------------------------------------------------- roll gap

def roll_gaps(fut_daily: pd.DataFrame, index_daily: pd.DataFrame) -> dict:
    """The jump in the continuous series at each expiry inside the data."""
    f = fut_daily.assign(day=_vn_date(fut_daily["open_time"]))[["day", "close"]].rename(columns={"close": "fut"})
    s = index_daily.assign(day=_vn_date(index_daily["open_time"]))[["day", "close"]].rename(columns={"close": "idx"})
    both = f.merge(s, on="day").sort_values("day").reset_index(drop=True)
    both["basis"] = both["fut"] - both["idx"]
    rows = []
    if len(both) >= 2:
        first, last = both["day"].iloc[0], both["day"].iloc[-1]
        y, m = first.year, first.month
        while date(y, m, 1) <= last:
            e = expiry(y, m)
            on = both[both["day"] == e]
            after = both[both["day"] > e].head(1)
            if not on.empty and not after.empty:
                gap = float(after["basis"].iloc[0] - on["basis"].iloc[0])
                rows.append({"expiry": e.isoformat(), "next_session": after["day"].iloc[0].isoformat(),
                             "basis_on_expiry": float(on["basis"].iloc[0]),
                             "basis_after": float(after["basis"].iloc[0]), "gap": gap,
                             "gap_pct": gap / float(on["idx"].iloc[0]) * 100})
            y, m = _next_month(y, m)
    gaps = np.array([r["gap"] for r in rows], dtype="float64")
    return {"rolls": rows, "summary": _stats(gaps) if rows else {"n": 0}}


# ------------------------------------------------------------------ sessions

def sessions(fut: pd.DataFrame) -> dict:
    """Each day split into its parts, in points; see the module doc."""
    df = fut.copy()
    df["day"] = _vn_date(df["open_time"])
    df["mod"] = _minute_of_day(df["open_time"])

    def part(lo, hi):
        p = df[(df["mod"] >= lo) & (df["mod"] < hi)].groupby("day")
        return pd.DataFrame({"open": p["open"].first(), "close": p["close"].last(),
                             "high": p["high"].max(), "low": p["low"].min(), "volume": p["volume"].sum()})

    am, pm, atc = part(*AM), part(*PM), part(*ATC)
    days = sorted(set(am.index) & set(pm.index))
    total_volume = df.groupby("day")["volume"].sum()
    out = {}
    if not days:
        return {"measured": False, "days": 0}
    am, pm = am.loc[days], pm.loc[days]
    day_close = df.groupby("day")["close"].last()
    prev_close = day_close.shift(1).reindex(days)

    moves = {
        "overnight": (am["open"] - prev_close).to_numpy(dtype="float64"),
        "morning": (am["close"] - am["open"]).to_numpy(dtype="float64"),
        "lunch": (pm["open"] - am["close"]).to_numpy(dtype="float64"),
        "afternoon": (pm["close"] - pm["open"]).to_numpy(dtype="float64"),
    }
    if not atc.empty:
        atc_days = [d for d in days if d in atc.index]
        moves["atc"] = (atc.loc[atc_days, "close"] - pm.loc[atc_days, "close"]).to_numpy(dtype="float64")
    for name, values in moves.items():
        out[name] = _stats(values)
    for name, frame in (("morning", am), ("afternoon", pm)):
        out[name]["range_mean"] = float((frame["high"] - frame["low"]).mean())
        vol = total_volume.reindex(days)
        out[name]["volume_share"] = float((frame["volume"] / vol).mean()) if (vol > 0).all() else None
    if "atc" in out and not atc.empty:
        vol = total_volume.reindex(atc.index)
        out["atc"]["volume_share"] = float((atc["volume"] / vol).mean()) if (vol > 0).all() else None
    same = np.sign(moves["morning"]) == np.sign(moves["afternoon"])
    nonzero = (moves["morning"] != 0) & (moves["afternoon"] != 0)
    return {"measured": True, "days": len(days), "parts": out,
            "same_direction_share": float(same[nonzero].mean()) if nonzero.any() else None,
            "tests": len(moves)}


def overview(fut_1m: pd.DataFrame, index_1m: pd.DataFrame, fut_1d: pd.DataFrame,
             index_1d: pd.DataFrame, today: date | None = None) -> dict:
    today = today or (datetime.now(timezone.utc) + timedelta(hours=7)).date()
    return {"calendar": listed(today), "basis": basis(fut_1m, index_1m),
            "rolls": roll_gaps(fut_1d, index_1d), "sessions": sessions(fut_1m),
            "today": today.isoformat()}
