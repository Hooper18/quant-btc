"""技术指标计算引擎。

输入输出统一为 Polars DataFrame；内部用 pandas-ta 计算后再转回 Polars。

列命名规范：所有列名都包含参数，避免歧义：
- `sma_20`, `ema_50`, `rsi_14`, `atr_14`
- `macd_line_12_26_9`, `macd_signal_12_26_9`, `macd_histogram_12_26_9`
- `bb_upper_20_2.0`, `bb_middle_20_2.0`, `bb_lower_20_2.0`
- `adx_14`, `dmp_14`, `dmn_14`
- `stoch_k_14_3_3`, `stoch_d_14_3_3`
- `kc_upper_20_10_2.0`, `kc_middle_20_10_2.0`, `kc_lower_20_10_2.0`
- `cci_20`, `williams_r_14`, `mfi_14`
- `obv`, `vwap`, `cmf_20`

策略 YAML 引用列名时必须使用上述带参数的形式。
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_ta as ta
import polars as pl


def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """转 pandas，保持 timestamp 为列。"""
    return df.to_pandas()


def _attach(df: pl.DataFrame, **new_cols: pl.Series | list) -> pl.DataFrame:
    """在 Polars DataFrame 上追加列；自动跳过 None。"""
    series_list: list[pl.Series] = []
    for name, vals in new_cols.items():
        if vals is None:
            continue
        if isinstance(vals, pl.Series):
            series_list.append(vals.alias(name))
        else:
            series_list.append(pl.Series(name, vals))
    if not series_list:
        return df
    return df.with_columns(series_list)


def _pdseries_to_pl(s: pd.Series | None, name: str, n_expected: int | None = None) -> pl.Series:
    """pandas Series → Polars Series；兼容 pandas-ta 在边界参数下返回 None / 空 / 短于预期长度。"""
    if s is None or len(s) == 0:
        if n_expected is None:
            return pl.Series(name, [], dtype=pl.Float64)
        return pl.Series(name, [None] * n_expected, dtype=pl.Float64)
    vals = s.astype("float64").tolist()
    if n_expected is not None and len(vals) < n_expected:
        vals = [None] * (n_expected - len(vals)) + vals  # type: ignore[list-item]
    return pl.Series(name, vals, dtype=pl.Float64)


class IndicatorEngine:
    """对一份 OHLCV Polars DataFrame 链式追加技术指标列。

    输入需含列：timestamp, open, high, low, close, volume。
    每个方法返回新的 DataFrame（原 df 不变），方便链式调用。
    """

    REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, df: pl.DataFrame):
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame 缺列：{missing}")
        self.df = df.sort("timestamp")
        self.n = self.df.height
        self._pd: pd.DataFrame | None = None

    @property
    def pd(self) -> pd.DataFrame:
        if self._pd is None:
            self._pd = _to_pandas(self.df)
        return self._pd

    def _pl(self, s: pd.Series | None, name: str = "") -> pl.Series:
        """pandas → polars；自动按 self.n 对齐（处理 pandas-ta 在边界参数下的 None/空返回）。"""
        return _pdseries_to_pl(s, name, n_expected=self.n)

    # ---------- 趋势类 ----------
    def sma(self, period: int = 20) -> pl.DataFrame:
        # period=1 SMA 数学上等于 close；pandas-ta 在 length=1 会抛 ValueError，绕开
        if period <= 1:
            return _attach(self.df, **{f"sma_{period}": self.df["close"].cast(pl.Float64)})
        out = ta.sma(self.pd["close"], length=period)
        return _attach(self.df, **{f"sma_{period}": self._pl(out, f"sma_{period}")})

    def ema(self, period: int = 20) -> pl.DataFrame:
        out = ta.ema(self.pd["close"], length=period)
        return _attach(self.df, **{f"ema_{period}": self._pl(out, f"ema_{period}")})

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
        out = ta.macd(self.pd["close"], fast=fast, slow=slow, signal=signal)
        suffix = f"{fast}_{slow}_{signal}"
        return _attach(
            self.df,
            **{
                f"macd_line_{suffix}": self._pl(out[f"MACD_{suffix}"], ""),
                f"macd_histogram_{suffix}": self._pl(out[f"MACDh_{suffix}"], ""),
                f"macd_signal_{suffix}": self._pl(out[f"MACDs_{suffix}"], ""),
            },
        )

    def adx(self, period: int = 14) -> pl.DataFrame:
        out = ta.adx(self.pd["high"], self.pd["low"], self.pd["close"], length=period)
        return _attach(
            self.df,
            **{
                f"adx_{period}": self._pl(out[f"ADX_{period}"], ""),
                f"dmp_{period}": self._pl(out[f"DMP_{period}"], ""),
                f"dmn_{period}": self._pl(out[f"DMN_{period}"], ""),
            },
        )

    # ---------- 动量类 ----------
    def rsi(self, period: int = 14) -> pl.DataFrame:
        out = ta.rsi(self.pd["close"], length=period)
        return _attach(self.df, **{f"rsi_{period}": self._pl(out, "")})

    def stoch(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> pl.DataFrame:
        out = ta.stoch(
            self.pd["high"], self.pd["low"], self.pd["close"],
            k=k_period, d=d_period, smooth_k=smooth_k,
        )
        suffix = f"{k_period}_{d_period}_{smooth_k}"
        return _attach(
            self.df,
            **{
                f"stoch_k_{suffix}": self._pl(out[f"STOCHk_{suffix}"], ""),
                f"stoch_d_{suffix}": self._pl(out[f"STOCHd_{suffix}"], ""),
            },
        )

    def cci(self, period: int = 20) -> pl.DataFrame:
        out = ta.cci(self.pd["high"], self.pd["low"], self.pd["close"], length=period)
        return _attach(self.df, **{f"cci_{period}": self._pl(out, "")})

    def williams_r(self, period: int = 14) -> pl.DataFrame:
        out = ta.willr(self.pd["high"], self.pd["low"], self.pd["close"], length=period)
        return _attach(self.df, **{f"williams_r_{period}": self._pl(out, "")})

    def mfi(self, period: int = 14) -> pl.DataFrame:
        out = ta.mfi(
            self.pd["high"], self.pd["low"], self.pd["close"], self.pd["volume"], length=period
        )
        return _attach(self.df, **{f"mfi_{period}": self._pl(out, "")})

    # ---------- 波动率类 ----------
    def bollinger(self, period: int = 20, std_dev: float = 2.0) -> pl.DataFrame:
        out = ta.bbands(self.pd["close"], length=period, std=std_dev)
        # pandas-ta 列名形如 BBL_20_2.0_2.0（最后那个 2.0 是 ddof）
        bbl = next(c for c in out.columns if c.startswith(f"BBL_{period}_{std_dev}"))
        bbm = next(c for c in out.columns if c.startswith(f"BBM_{period}_{std_dev}"))
        bbu = next(c for c in out.columns if c.startswith(f"BBU_{period}_{std_dev}"))
        suffix = f"{period}_{std_dev}"
        return _attach(
            self.df,
            **{
                f"bb_lower_{suffix}": self._pl(out[bbl], ""),
                f"bb_middle_{suffix}": self._pl(out[bbm], ""),
                f"bb_upper_{suffix}": self._pl(out[bbu], ""),
            },
        )

    def atr(self, period: int = 14) -> pl.DataFrame:
        out = ta.atr(self.pd["high"], self.pd["low"], self.pd["close"], length=period)
        return _attach(self.df, **{f"atr_{period}": self._pl(out, "")})

    def keltner(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pl.DataFrame:
        out = ta.kc(
            self.pd["high"], self.pd["low"], self.pd["close"],
            length=period, atr_length=atr_period, scalar=multiplier,
        )
        # pandas-ta 列名形如 KCLe_20_2 / KCBe_20_2 / KCUe_20_2，乘数取整数后无小数
        mul_token = int(multiplier) if float(multiplier).is_integer() else multiplier
        kcl = next(c for c in out.columns if c.startswith("KCLe_"))
        kcb = next(c for c in out.columns if c.startswith("KCBe_"))
        kcu = next(c for c in out.columns if c.startswith("KCUe_"))
        suffix = f"{period}_{atr_period}_{mul_token}"
        return _attach(
            self.df,
            **{
                f"kc_lower_{suffix}": self._pl(out[kcl], ""),
                f"kc_middle_{suffix}": self._pl(out[kcb], ""),
                f"kc_upper_{suffix}": self._pl(out[kcu], ""),
            },
        )

    # ---------- 成交量类 ----------
    def obv(self) -> pl.DataFrame:
        out = ta.obv(self.pd["close"], self.pd["volume"])
        return _attach(self.df, obv=self._pl(out, ""))

    def vwap(self) -> pl.DataFrame:
        # pandas-ta 的 vwap 需要 DatetimeIndex
        pdf = self.pd.copy()
        pdf.index = pd.DatetimeIndex(pdf["timestamp"])
        out = ta.vwap(pdf["high"], pdf["low"], pdf["close"], pdf["volume"])
        return _attach(self.df, vwap=self._pl(out.reset_index(drop=True), ""))

    def cmf(self, period: int = 20) -> pl.DataFrame:
        out = ta.cmf(
            self.pd["high"], self.pd["low"], self.pd["close"], self.pd["volume"], length=period
        )
        return _attach(self.df, **{f"cmf_{period}": self._pl(out, "")})

    # ---------- Phase11 衍生指标（依赖额外数据源）----------
    def taker_buy_ratio(self) -> pl.DataFrame:
        """主动买入占比 = taker_buy_volume / volume。

        值域 [0, 1]：>0.5 表示买盘主动，<0.5 表示卖盘主动。
        需要 DataFrame 含 `taker_buy_volume` 列（来自 Phase11 后的 OHLCV downloader）。
        """
        if "taker_buy_volume" not in self.df.columns:
            raise KeyError(
                "taker_buy_ratio 需要 'taker_buy_volume' 列；请确认 OHLCV parquet "
                "是 Phase11 之后下载的（旧版 schema 缺该列）"
            )
        return self.df.with_columns(
            pl.when(pl.col("volume") > 0)
                .then(pl.col("taker_buy_volume") / pl.col("volume"))
                .otherwise(None)
                .alias("taker_buy_ratio")
        )

    def oi_change(self, period: int = 1) -> pl.DataFrame:
        """持仓量 N 周期变化率（百分比）= (OI[t] / OI[t-period] - 1) × 100。

        例：返回 5.0 表示过去 N 根 K 线 OI 上涨 5%；策略 YAML 阈值用 ±5 这种直观数字。
        需要 DataFrame 含 `open_interest` 列（通过 data_merger.merge_market_data 注入）。
        """
        if "open_interest" not in self.df.columns:
            raise KeyError(
                "oi_change 需要 'open_interest' 列；先用 data_merger.merge_market_data "
                "把 OI 合并到 OHLCV"
            )
        col_name = f"oi_change_{period}"
        return self.df.with_columns(
            ((pl.col("open_interest") / pl.col("open_interest").shift(period) - 1.0) * 100.0)
            .alias(col_name)
        )

    def rolling_max(self, period: int = 20) -> pl.DataFrame:
        """过去 N 根 K 线（不含当前）的最高 close。

        实现 = `close.shift(1).rolling_max(period)`；这样
        ``close > rolling_max_N`` 即"创 N 周期新高"，自身不会"自我证明"。
        """
        col_name = f"rolling_max_{period}"
        return self.df.with_columns(
            pl.col("close").shift(1).rolling_max(period).alias(col_name)
        )

    def rolling_min(self, period: int = 20) -> pl.DataFrame:
        """过去 N 根 K 线（不含当前）的最低 close。配合 ``close < rolling_min_N`` 表"创新低"。"""
        col_name = f"rolling_min_{period}"
        return self.df.with_columns(
            pl.col("close").shift(1).rolling_min(period).alias(col_name)
        )

    def fear_greed_ma(self, period: int = 7) -> pl.DataFrame:
        """恐慌贪婪指数移动平均（period 个 bar 滚动均值）。

        需要 DataFrame 含 `fear_greed` 列（通过 data_merger.merge_market_data 注入；
        FGI 日频，会被 forward-fill 到每根 K 线）。
        """
        if "fear_greed" not in self.df.columns:
            raise KeyError(
                "fear_greed_ma 需要 'fear_greed' 列；先用 data_merger.merge_market_data "
                "把 FGI 合并到 OHLCV"
            )
        col_name = f"fear_greed_ma_{period}"
        return self.df.with_columns(
            pl.col("fear_greed").cast(pl.Float64).rolling_mean(period).alias(col_name)
        )

    # ---------- 批量 ----------
    def compute_all(
        self,
        config: dict[str, dict[str, Any]] | list[tuple[str, dict[str, Any]]],
    ) -> pl.DataFrame:
        """按配置批量计算指标，逐次合并到一份 DataFrame。

        config 支持两种格式：
        - dict 形式（每种指标只用一组参数）：
          ``{"rsi": {"period": 14}, "macd": {"fast": 12, "slow": 26, "signal": 9}}``
        - list[tuple] 形式（同一指标可有多组参数，例如 ema_12 + ema_26）：
          ``[("ema", {"period": 12}), ("ema", {"period": 26})]``
        未知 key 会抛 KeyError，避免拼写错误静默失败。
        """
        method_map = {
            "sma": self.sma, "ema": self.ema, "macd": self.macd, "adx": self.adx,
            "rsi": self.rsi, "stoch": self.stoch, "cci": self.cci,
            "williams_r": self.williams_r, "mfi": self.mfi,
            "bollinger": self.bollinger, "atr": self.atr, "keltner": self.keltner,
            "obv": self.obv, "vwap": self.vwap, "cmf": self.cmf,
            "taker_buy_ratio": self.taker_buy_ratio,
            "oi_change": self.oi_change,
            "fear_greed_ma": self.fear_greed_ma,
            "rolling_max": self.rolling_max,
            "rolling_min": self.rolling_min,
        }
        items = list(config.items()) if isinstance(config, dict) else list(config)
        result = self.df
        for name, params in items:
            if name not in method_map:
                raise KeyError(f"未知指标 {name}；可选：{sorted(method_map)}")
            params = params or {}
            # 用临时引擎承载已累积的列，确保下一次基于最新结果
            tmp = IndicatorEngine.__new__(IndicatorEngine)
            tmp.df = result
            tmp.n = result.height
            tmp._pd = None
            new_df = method_map[name].__func__(tmp, **params)
            # 合并新增列（避免重复）
            new_cols = [c for c in new_df.columns if c not in result.columns]
            if new_cols:
                result = result.with_columns([new_df[c] for c in new_cols])
        return result


# ---------- 交叉检测 ----------
def crossover(a: pl.Series, b: pl.Series) -> pl.Series:
    """A 上穿 B：当前 bar a>b 且前一 bar a<=b。首行返回 False。"""
    cur = a > b
    prev = a.shift(1) <= b.shift(1)
    return (cur & prev).fill_null(False)


def crossunder(a: pl.Series, b: pl.Series) -> pl.Series:
    """A 下穿 B：当前 bar a<b 且前一 bar a>=b。首行返回 False。"""
    cur = a < b
    prev = a.shift(1) >= b.shift(1)
    return (cur & prev).fill_null(False)
