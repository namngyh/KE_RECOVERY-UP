"""S8 - dựng tín hiệu cho M0-M9 bằng CHÍNH các hàm của mã KE đã đóng băng (strategies/ke_recovery.py).

build_signal(lab, spec) tái tạo ke.signals() có thêm các "móc" của đề cương:
  * g^{Gk} = g0 + lambda * x~          (mục 6.3, M6; M1 dùng x = log h~ theo ngày)
  * chuẩn hóa theo lịch cho RSI/%K      (M5B)           * tín hiệu trên chuỗi hiệu chỉnh (M5A)
  * bộ lọc CHỈ chặn lệnh mở mới         (M2, M3, M4, M5C, M7: đề cương 8.2 'theo dõi đầy đủ vị thế đang mở')
  * bộ lọc LinReg/WMA thích nghi theo ER (M8)
Với spec rỗng kết quả PHẢI trùng ke.signals() từng nến (được kiểm tra bằng test và cổng của pipeline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies import ke_recovery as ke

from . import calendar_feats as cf
from .prereg import PREREG, period_of

DEFAULTS = {k: v["default"] for k, v in ke.STRATEGY["params"].items()}


class Lab:
    """Gói dữ liệu cho mọi mô hình: giá thực (khớp lệnh/P&L), giá hiệu chỉnh (tín hiệu), đặc trưng, lịch, dự báo EGARCH-X."""

    def __init__(self, raw: pd.DataFrame, adj: pd.DataFrame, feats: pd.DataFrame, barcal: pd.DataFrame | None = None,
                 egarch_daily: pd.DataFrame | None = None):
        self.raw = raw.reset_index(drop=True)          # cột: ts, open_time_ms, open, high, low, close
        self.adj = adj.reset_index(drop=True)          # cùng cột nhưng giá đã hiệu chỉnh roll
        self.F = feats.reset_index(drop=True)
        self.bc = barcal.reset_index(drop=True) if barcal is not None else None
        self.n = len(self.raw)
        self.period = period_of(self.raw["ts"]).to_numpy()
        self.dev = self.period == "dev"
        self.day = self.raw["ts"].dt.normalize()
        self.slot = cf.slot_minute(self.raw["ts"])
        self.eg = egarch_daily
        self.meta_p: tuple[np.ndarray, np.ndarray] | None = None
        self.diag: dict = {}
        self._cache: dict = {}

    # ---- giá và thành phần của KE
    def price_frame(self, kind: str) -> pd.DataFrame:
        return self.raw if kind == "raw" else self.adj

    def rsi_k(self, kind: str, period: int, k_smooth: int):
        key = ("rsik", kind, period, k_smooth)
        if key not in self._cache:
            d = self.price_frame(kind)
            self._cache[key] = (ke._rsi(d["close"], period),
                                ke._stoch_k(d["high"], d["low"], d["close"], period, k_smooth))
        return self._cache[key]

    def dev_std(self, s: pd.Series) -> float:
        return float(pd.Series(np.asarray(s))[self.dev].std())

    def g0(self, kind: str = "raw", params: dict | None = None) -> pd.Series:
        p = {**DEFAULTS, **(params or {})}
        rsi, k = self.rsi_k(kind, p["period"], p["k_smooth"])
        return p["n1"] * rsi + k

    def g0_sd_dev(self) -> float:
        """std(g0) trên dev: thang để đặt lambda = c * std(g0) (prereg)."""
        if "g0sd" not in self._cache:
            self._cache["g0sd"] = self.dev_std(self.g0("raw"))
        return self._cache["g0sd"]

    # ---- đặc trưng theo nến, theo ngày
    def feature(self, col: str) -> pd.Series:
        if col == "logh_z":
            return pd.Series(self.logh_z_bar(), index=self.raw.index)
        return self.F[col]

    def _daily_to_bar(self, daily: pd.Series) -> np.ndarray:
        return self.day.map(daily).to_numpy(dtype=float)

    def logh_z_bar(self) -> np.ndarray:
        """z-score mở rộng (chỉ quá khứ, tối thiểu 60 phiên) của log h^_{d|d-1}, gán cho mọi nến của phiên d. Thiếu -> NaN."""
        if "logh_z" not in self._cache:
            if self.eg is None:
                self._cache["logh_z"] = np.full(self.n, np.nan)
            else:
                lh = np.log(self.eg["h_forecast"])
                past_mean = lh.shift(1).expanding(min_periods=60).mean()
                past_std = lh.shift(1).expanding(min_periods=60).std(ddof=1)
                self._cache["logh_z"] = self._daily_to_bar((lh - past_mean) / past_std)
        return self._cache["logh_z"]

    def vol_pct_bar(self) -> np.ndarray:
        """Phân vị mở rộng (chỉ quá khứ) của h^_{d|d-1} so với các dự báo trước đó (đề cương 8.2: Q^sigma_d ước lượng trước phiên d)."""
        if "vol_pct" not in self._cache:
            if self.eg is None:
                self._cache["vol_pct"] = np.full(self.n, np.nan)
            else:
                h = self.eg["h_forecast"].to_numpy(dtype=float)
                pct = np.full(len(h), np.nan)
                for i in range(len(h)):
                    if not np.isfinite(h[i]):
                        continue
                    past = h[:i][np.isfinite(h[:i])]            # chỉ các dự báo hữu hạn TRƯỚC phiên i
                    if len(past) >= 60:
                        pct[i] = np.mean(past <= h[i])
                self._cache["vol_pct"] = self._daily_to_bar(pd.Series(pct, index=self.eg.index))
        return self._cache["vol_pct"]

    def er_threshold(self, q: float) -> float:
        return float(self.F["ER"][self.dev].quantile(q))

    def er_regime(self, quantiles) -> tuple[np.ndarray, np.ndarray]:
        """Chế độ ER (0 thấp, 1 giữa, 2 cao) cho mọi nến của phiên d, lấy từ ER TRUNG VỊ của phiên d-1 (cập nhật mỗi phiên,
        không đổi trong phiên). Ngưỡng là các phân vị của ER trung vị phiên trên dev. valid=False khi chưa có phiên trước."""
        key = ("er_regime", tuple(quantiles))
        if key not in self._cache:
            med = self.F["ER"].groupby(self.day.to_numpy()).median()          # theo phiên
            dev_days = pd.Index(self.day[self.dev].unique())
            lo_t, hi_t = (float(med.loc[med.index.isin(dev_days)].quantile(q)) for q in quantiles)
            prev = med.shift(1)                                                # chỉ dùng phiên trước
            per_bar = self.day.map(prev).to_numpy(dtype=float)
            valid = ~np.isnan(per_bar)
            regime = np.where(per_bar < lo_t, 0, np.where(per_bar < hi_t, 1, 2))
            self._cache[key] = (regime, valid)
        return self._cache[key]


def build_gates(lab: Lab, gates: list[dict] | None):
    """Trả về (up_extra, dn_extra): mảng bool chỉ chặn lệnh MỞ MỚI; None nếu không có bộ lọc."""
    if not gates:
        return None, None
    n = lab.n
    up = np.ones(n, bool); dn = np.ones(n, bool)
    for g in gates:
        t = g["t"]
        if t in ("vol_band", "vol_upper"):
            pct = lab.vol_pct_bar()
            lo = g.get("lo", 0.0); hi = g["hi"]
            ok = np.where(np.isnan(pct), True, (pct >= lo) & (pct <= hi))   # chưa có dự báo -> không lọc
            up &= ok; dn &= ok
        elif t == "er_min":
            thr = lab.er_threshold(g["q"])
            er = lab.F["ER"].to_numpy(dtype=float)
            ok = np.where(np.isnan(er), False, er >= thr)
            up &= ok; dn &= ok
        elif t == "cal":
            bc = lab.bc
            near = bc["near_expiry"].to_numpy(bool)
            hol = (bc["after_holiday"].to_numpy(bool) | bc["first_day_after_expiry"].to_numpy(bool))
            v = g["variant"]
            ok = ~near if v == "no_entry_near_expiry" else ~hol if v == "no_entry_after_holiday_or_first_day" else ~(near | hol)
            up &= ok; dn &= ok
        elif t == "meta":
            if lab.meta_p is None:
                raise ValueError("Chưa có xác suất meta-label")
            pl, ps = lab.meta_p
            up &= np.where(np.isnan(pl), True, pl >= g["tau"])
            dn &= np.where(np.isnan(ps), True, ps >= g["tau"])
        else:
            raise ValueError(f"Bộ lọc không biết: {t}")
    return up, dn


def build_signal(lab: Lab, spec: dict | None = None) -> np.ndarray:
    """Tín hiệu vị thế mục tiêu {-1,0,1} theo nến (khớp ở open nến kế tiếp do engine xử lý)."""
    spec = spec or {}
    p = {**DEFAULTS, **spec.get("params", {})}
    kind = spec.get("price", "raw")
    df = lab.price_frame(kind)
    close = df["close"]
    rsi, k = lab.rsi_k(kind, p["period"], p["k_smooth"])
    mode = spec.get("g_mode", "ke")
    if mode == "ke":
        g = p["n1"] * rsi + k
    elif mode == "rsi_only":
        g = p["n1"] * rsi
    elif mode == "k_only":
        g = k
    elif mode == "m5b":  # M5B: RSI và %K chuẩn hóa theo lịch, giữ tỷ trọng tương đối của KE gốc (weight từ std trên dev)
        f = PREREG["features"]
        rt = cf.slot_standardize(rsi, lab.slot, f["slot_window"], f["slot_min_periods"], f["eps0"])
        kt = cf.slot_standardize(k, lab.slot, f["slot_window"], f["slot_min_periods"], f["eps0"])
        g = (p["n1"] * lab.dev_std(rsi)) * rt.fillna(0.0) + lab.dev_std(k) * kt.fillna(0.0)
        g = g.where(rsi.notna() & k.notna())
    else:
        raise ValueError(f"g_mode không biết: {mode}")
    for col, lam in spec.get("add", []):
        g = g + lam * lab.feature(col).fillna(0.0)

    up_extra, dn_extra = build_gates(lab, spec.get("gates"))
    ema_prev = ke._ema(close, p["ema_length"]).shift(1).to_numpy()
    c = close.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        up_gate = (c > ema_prev) if spec.get("ema_gate", True) else np.ones(lab.n, bool)
        dn_gate = (c < ema_prev) if spec.get("ema_gate", True) else np.ones(lab.n, bool)
    if up_extra is not None:
        up_gate = up_gate & up_extra
        dn_gate = dn_gate & dn_extra

    if spec.get("adaptive"):
        m8 = PREREG["M8"]
        regime, valid = lab.er_regime(m8["regimes_er_q"])
        fs, ss = [], []
        for name in ("low", "mid", "high"):
            L, m = m8["L_m"][name]
            f_r = ke._linreg(g, L)
            fs.append(f_r.to_numpy()); ss.append(ke._wma(f_r, m).to_numpy())
        f_arr = np.choose(regime, fs); s_arr = np.choose(regime, ss)
        f_arr = np.where(valid, f_arr, np.nan)
        buy0 = ke._cross(f_arr, s_arr); short0 = ke._cross(s_arr, f_arr)
        change = np.concatenate([[False], regime[1:] != regime[:-1]])
        lab.diag["M8_crosses_raw"] = int((buy0 | short0).sum())
        lab.diag["M8_crosses_at_regime_change"] = int(((buy0 | short0) & change).sum())
        buy0 = buy0 & ~change; short0 = short0 & ~change     # không ghi nhận giao cắt tại nến đổi chế độ (prereg)
    else:
        smoother = spec.get("smoother", "linreg")
        if smoother == "linreg":
            f_s = ke._linreg(g, p["reg_period"])
        elif smoother == "wma":
            f_s = ke._wma(g, p["reg_period"])
        elif smoother == "sma":
            f_s = g.rolling(p["reg_period"]).mean()
        else:
            raise ValueError(f"smoother không biết: {smoother}")
        s_s = ke._wma(f_s, p["ma_period"])
        buy0 = ke._cross(f_s, s_s); short0 = ke._cross(s_s, f_s)

    pos = ke._positions(ke._flip(buy0, short0), ke._flip(short0, buy0), buy0, short0, up_gate, dn_gate)
    pos = pos.astype(int)
    side = spec.get("side")
    if side == "long":
        pos = np.where(pos > 0, pos, 0)
    elif side == "short":
        pos = np.where(pos < 0, pos, 0)
    if spec.get("flat_eod"):  # A7: đóng vị thế ở 2 nến cuối mỗi phiên (khớp ở giá ATC), mở lại ở phiên sau
        last2 = lab.raw.groupby(lab.day).cumcount(ascending=False).to_numpy() < 2
        pos = np.where(last2, 0, pos)
    return pos
