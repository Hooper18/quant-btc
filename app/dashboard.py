"""Streamlit 交互式回测仪表盘。

启动：uv run streamlit run app/dashboard.py

页面：
1. 策略回测：参数滑块 + 一键回测 + 净值/月度热力图/交易表
2. 策略对比：多选策略叠加净值
3. 数据浏览：K 线 + 可勾选指标 + 市场情绪
4. 风险分析：蒙特卡洛 + 杠杆扫描

设计：
- 大对象（aux 数据、OHLCV+合并+指标）用 st.cache_resource 缓存，避免每次操作重算
- 回测结果按 (yaml 内容 + bt 配置 + 数据 hash) 缓存
- 所有图表用 plotly（交互、可缩放、深色主题）
"""
from __future__ import annotations

import copy
import re
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import polars as pl
import streamlit as st
import yaml
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester, BacktestResult, MonteCarloSimulator  # noqa: E402
from backtest.optimizer import set_param  # noqa: E402
from indicators import IndicatorEngine  # noqa: E402
from utils.config import DataConfig  # noqa: E402

from run_backtest import (  # noqa: E402
    _collect_required_indicators,
    build_data_dict,
    load_aux_data,
)


# ---------- 常量与缓存 ----------
DATA_CFG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
BT_CFG_PATH = PROJECT_ROOT / "config" / "backtest_config.yaml"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data" / "parquet" / "BTCUSDT"


@st.cache_resource(show_spinner="加载市场数据…")
def get_data_cfg() -> DataConfig:
    return DataConfig.from_yaml(DATA_CFG_PATH)


@st.cache_resource(show_spinner="加载辅助数据 (funding/OI/FGI/LS/TT)…")
def get_aux() -> dict[str, pl.DataFrame | None]:
    return load_aux_data(get_data_cfg())


@st.cache_resource(show_spinner="加载 OHLCV 并合并辅助数据…")
def get_data_dict_for(used_tfs_tuple: tuple[str, ...], ind_cfg_key: str) -> dict[str, pl.DataFrame]:
    """缓存 build_data_dict 结果。
    cache key 由 used_tfs_tuple + ind_cfg_key 确定（ind_cfg_key 是 indicator 配置的字符串表示）。"""
    cfg = get_data_cfg()
    aux = get_aux()
    ind_cfg = _ind_cfg_from_key(ind_cfg_key)
    return build_data_dict(cfg, list(used_tfs_tuple), ind_cfg, aux=aux)


def _ind_cfg_to_key(ind_cfg: list[tuple[str, dict[str, Any]]]) -> str:
    return repr(sorted((k, sorted(v.items())) for k, v in ind_cfg))


def _ind_cfg_from_key(key: str) -> list[tuple[str, dict[str, Any]]]:
    raw = eval(key, {"__builtins__": {}}, {})
    return [(name, dict(items)) for name, items in raw]


@st.cache_resource(show_spinner="跑回测…")
def run_backtest_cached(
    yaml_text: str,
    bt_params: tuple,        # hashable tuple of (init, lev, fee, slip, mmr, dd, primary_tf, daily_loss)
    used_tfs_tuple: tuple[str, ...],
    ind_cfg_key: str,
    has_funding: bool,
) -> BacktestResult:
    bt = Backtester(
        initial_balance=bt_params[0],
        leverage=bt_params[1],
        fee_rate=bt_params[2],
        slippage=bt_params[3],
        maintenance_margin_rate=bt_params[4],
        max_drawdown_pct=bt_params[5],
        funding_rate_epochs_utc=[0, 8, 16],
        primary_timeframe=bt_params[6],
        daily_max_loss_pct=bt_params[7],
    )
    data_dict = get_data_dict_for(used_tfs_tuple, ind_cfg_key)
    aux = get_aux()
    fr = aux.get("funding") if has_funding else None
    tmp = Path(tempfile.gettempdir()) / f"app_strat_{uuid.uuid4().hex}.yaml"
    tmp.write_text(yaml_text, encoding="utf-8")
    try:
        return bt.run(data_dict, str(tmp), funding_rate_df=fr)
    finally:
        tmp.unlink(missing_ok=True)


# ---------- 工具：从 YAML 推 TF ----------
def used_timeframes(strategies: list[dict], primary: str) -> set[str]:
    used = {primary}
    for s in strategies:
        for c in s.get("conditions", []):
            stack = [c]
            while stack:
                cur = stack.pop()
                if "conditions" in cur:
                    stack.extend(cur["conditions"])
                tf = cur.get("timeframe")
                if tf:
                    used.add(tf)
    return used


def list_preset_strategies() -> dict[str, Path]:
    out = {}
    for f in sorted(CONFIG_DIR.glob("strategies*.yaml")):
        out[f.stem] = f
    # 也加入 v2
    v2 = CONFIG_DIR / "strategies_v2_optimized.yaml"
    if v2.exists():
        out["strategies_v2_optimized"] = v2
    return dict(sorted(out.items()))


def load_bt_cfg() -> dict:
    with BT_CFG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- 页面 1：策略回测 ----------
def page_backtest() -> None:
    st.title("📈 策略回测")
    st.caption("修改参数 → 实时跑回测 → 看净值 / 指标 / 月度热力图 / 交易明细")

    presets = list_preset_strategies()

    with st.sidebar:
        st.subheader("策略来源")
        src_mode = st.radio("选择方式", ["预置策略", "上传 YAML"], horizontal=True)
        if src_mode == "预置策略":
            choice = st.selectbox("策略", list(presets.keys()),
                                  index=list(presets.keys()).index("strategies_v2_optimized")
                                  if "strategies_v2_optimized" in presets else 0)
            base_yaml_text = presets[choice].read_text(encoding="utf-8")
        else:
            uploaded = st.file_uploader("上传策略 YAML", type=["yaml", "yml"])
            base_yaml_text = uploaded.read().decode("utf-8") if uploaded else None
            if base_yaml_text is None:
                st.warning("请上传策略文件")
                return

        st.divider()
        st.subheader("参数微调（覆盖 YAML 第 1 个策略的相应字段）")
        st.caption("这些滑块会覆盖 strategies[0] 的参数；不改第 2 个策略")

        # 解析 base
        base_dict = yaml.safe_load(base_yaml_text)
        s0 = base_dict["strategies"][0]
        # 从 conditions 中找第一个 indicator=rsi_14 的 value
        cur_rsi_thr = None
        cur_rsi_path = None
        for i, c in enumerate(s0.get("conditions", [])):
            if c.get("indicator", "").startswith("rsi_") and "value" in c:
                cur_rsi_thr = c["value"]
                cur_rsi_path = f"strategies[0].conditions[{i}].value"
                break
        if cur_rsi_thr is not None:
            cur_rsi_thr = st.slider("RSI 阈值", 30, 95, int(cur_rsi_thr), 1)
        cur_size = st.slider("仓位 size_pct (%)", 1, 30,
                             int(s0.get("action", {}).get("size_pct", 5)), 1)
        cur_sl = st.slider("止损 stop_loss_pct (%)", 1, 10,
                           int(s0.get("stop_loss_pct", 5)), 1)
        cur_tp = st.slider("止盈 take_profit_pct (%)", 1, 15,
                           int(s0.get("take_profit_pct", 6)), 1)

        st.divider()
        st.subheader("回测引擎")
        bt_cfg_dict = load_bt_cfg()
        cur_lev = st.slider("杠杆", 1, 20,
                            int(bt_cfg_dict.get("leverage", 10)), 1,
                            help="Task1 杠杆扫描显示 7× 是破产 < 5% 安全上限")

        run_btn = st.button("🚀 运行回测", type="primary", use_container_width=True)

    if not run_btn:
        st.info("👈 配置参数后点击左下角 **运行回测**")
        return

    # 应用参数到 yaml
    mod = copy.deepcopy(base_dict)
    if cur_rsi_path is not None:
        set_param(mod, cur_rsi_path, cur_rsi_thr)
    set_param(mod, "strategies[0].action.size_pct", cur_size)
    set_param(mod, "strategies[0].stop_loss_pct", cur_sl)
    set_param(mod, "strategies[0].take_profit_pct", cur_tp)
    yaml_text = yaml.safe_dump(mod, allow_unicode=True, sort_keys=False)

    # 推算 TF / 指标
    primary_tf = bt_cfg_dict["primary_timeframe"]
    used_tfs = used_timeframes(mod["strategies"], primary_tf)
    ind_cfg = _collect_required_indicators(mod["strategies"])
    aux = get_aux()
    bt_params = (
        float(bt_cfg_dict["initial_balance"]),
        float(cur_lev),
        float(bt_cfg_dict["fee_rate"]),
        float(bt_cfg_dict["slippage"]),
        float(bt_cfg_dict["maintenance_margin_rate"]),
        float(bt_cfg_dict["max_drawdown_pct"]),
        primary_tf,
        bt_cfg_dict.get("daily_max_loss_pct"),
    )

    result = run_backtest_cached(
        yaml_text=yaml_text,
        bt_params=bt_params,
        used_tfs_tuple=tuple(sorted(used_tfs)),
        ind_cfg_key=_ind_cfg_to_key(ind_cfg),
        has_funding=aux.get("funding") is not None,
    )

    # 指标卡
    m = result.metrics
    cols = st.columns(5)
    cols[0].metric("总收益率", f"{m.get('total_return_pct', 0):+.2f}%")
    cols[1].metric("年化", f"{m.get('annualized_return_pct', 0):+.2f}%")
    cols[2].metric("夏普", f"{m.get('sharpe_ratio', 0):.3f}")
    cols[3].metric("最大回撤", f"{m.get('max_drawdown_pct', 0):.2f}%")
    cols[4].metric("胜率 / 交易数",
                   f"{m.get('win_rate_pct', 0):.1f}% / {int(m.get('total_trades', 0))}")
    if m.get("circuit_breaker"):
        st.error("⚠️ 触发熔断")

    # 净值 + 回撤
    st.subheader("净值曲线 + 回撤")
    eq = list(result.equity_curve)
    ts = list(result.timestamps)
    import numpy as np
    eq_arr = np.array(eq, dtype=float)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = np.where(peak > 0, (peak - eq_arr) / peak * 100, 0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=ts, y=eq, mode="lines", name="NAV", line=dict(color="#4cd2ff", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=dd_pct, fill="tozeroy", name="回撤 %",
                             line=dict(color="#ff6b6b", width=0.7)), row=2, col=1)
    fig.update_yaxes(title_text="净值 USDT", row=1, col=1)
    fig.update_yaxes(title_text="回撤 %", autorange="reversed", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=560, hovermode="x unified",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 月度热力图
    st.subheader("月度收益率热力图 (%)")
    monthly = _monthly_returns_table(result)
    if monthly is not None:
        fig_h = go.Figure(data=go.Heatmap(
            z=monthly.tolist(),
            x=list(range(1, 13)),
            y=list(range(_first_year(ts), _first_year(ts) + monthly.shape[0])),
            colorscale="RdYlGn", zmid=0,
            hovertemplate="月%{x} 年%{y}: %{z:.1f}%<extra></extra>",
            text=[[f"{v:.1f}" if v == v else "" for v in row] for row in monthly.tolist()],
            texttemplate="%{text}",
        ))
        fig_h.update_layout(template="plotly_dark", height=380,
                            xaxis_title="月", yaxis_title="年",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

    # 最近 30 笔交易
    st.subheader("最近 30 笔交易")
    if result.trades:
        rows = [{
            "时间": t.timestamp, "方向": t.side, "价格": round(t.price, 2),
            "数量": round(t.size, 5), "PnL": round(t.pnl, 2), "策略": t.strategy,
        } for t in result.trades[-30:]]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("无交易记录")


def _first_year(ts: list) -> int:
    return ts[0].year if ts else 2020


def _monthly_returns_table(result: BacktestResult):
    if not result.equity_curve or not result.timestamps:
        return None
    import numpy as np
    df = pl.DataFrame({"ts": result.timestamps, "eq": result.equity_curve}).sort("ts").with_columns(
        pl.col("ts").dt.year().alias("y"),
        pl.col("ts").dt.month().alias("m"),
    )
    monthly = (df.group_by(["y", "m"]).agg(pl.col("eq").last().alias("eq_end")).sort(["y", "m"]))
    years = sorted(set(monthly["y"].to_list()))
    if not years:
        return None
    ymin, ymax = min(years), max(years)
    years_full = list(range(ymin, ymax + 1))
    mat = np.full((len(years_full), 12), np.nan, dtype=float)
    prev = float(result.metrics.get("initial_balance", result.equity_curve[0]))
    for r in monthly.iter_rows(named=True):
        y, m, eq_end = int(r["y"]), int(r["m"]), float(r["eq_end"])
        mat[years_full.index(y), m - 1] = (eq_end - prev) / prev * 100 if prev > 0 else 0
        prev = eq_end
    return mat


# ---------- 页面 2：策略对比 ----------
def page_compare() -> None:
    st.title("📊 策略对比")
    st.caption("勾选多个策略 → 净值叠加 + 指标表对比")

    presets = list_preset_strategies()
    chosen = st.multiselect(
        "选择策略（≥ 2）",
        list(presets.keys()),
        default=[k for k in ["strategies", "strategies_v2_optimized"] if k in presets],
    )
    if len(chosen) < 1:
        st.warning("至少选 1 个策略")
        return
    if not st.button("▶️ 跑对比", type="primary"):
        return

    bt_cfg = load_bt_cfg()
    aux = get_aux()
    bt_params = (
        float(bt_cfg["initial_balance"]),
        float(bt_cfg["leverage"]),
        float(bt_cfg["fee_rate"]),
        float(bt_cfg["slippage"]),
        float(bt_cfg["maintenance_margin_rate"]),
        float(bt_cfg["max_drawdown_pct"]),
        bt_cfg["primary_timeframe"],
        bt_cfg.get("daily_max_loss_pct"),
    )

    results: list[tuple[str, BacktestResult]] = []
    progress = st.progress(0.0)
    for i, name in enumerate(chosen, 1):
        path = presets[name]
        text = path.read_text(encoding="utf-8")
        d = yaml.safe_load(text)
        used_tfs = used_timeframes(d["strategies"], bt_cfg["primary_timeframe"])
        ind_cfg = _collect_required_indicators(d["strategies"])
        try:
            r = run_backtest_cached(
                yaml_text=text, bt_params=bt_params,
                used_tfs_tuple=tuple(sorted(used_tfs)),
                ind_cfg_key=_ind_cfg_to_key(ind_cfg),
                has_funding=aux.get("funding") is not None,
            )
            results.append((name, r))
        except Exception as e:
            st.error(f"{name} 回测失败：{e}")
        progress.progress(i / len(chosen))

    if not results:
        st.error("无成功结果")
        return

    # 叠加净值
    fig = go.Figure()
    for name, r in results:
        fig.add_trace(go.Scatter(
            x=r.timestamps, y=r.equity_curve, mode="lines",
            name=name, line=dict(width=1.2),
        ))
    fig.update_layout(template="plotly_dark", height=520, hovermode="x unified",
                      yaxis_type="log", yaxis_title="净值 (USDT, 对数轴)",
                      margin=dict(l=10, r=10, t=20, b=10),
                      title="策略净值叠加")
    st.plotly_chart(fig, use_container_width=True)

    # 指标表
    rows = []
    for name, r in results:
        m = r.metrics
        rows.append({
            "策略": name,
            "总收益%": round(m.get("total_return_pct", 0), 2),
            "年化%": round(m.get("annualized_return_pct", 0), 2),
            "夏普": round(m.get("sharpe_ratio", 0), 3),
            "回撤%": round(m.get("max_drawdown_pct", 0), 2),
            "胜率%": round(m.get("win_rate_pct", 0), 2),
            "盈亏比": round(m.get("profit_loss_ratio", 0), 3),
            "交易数": int(m.get("total_trades", 0)),
            "熔断": "是" if m.get("circuit_breaker") else "否",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------- 页面 3：数据浏览 ----------
def page_data() -> None:
    st.title("🔍 数据浏览")
    st.caption("K 线 + 可勾选指标 + 市场情绪（FGI / 多空比 / OI）")

    cfg = get_data_cfg()
    aux = get_aux()

    col1, col2 = st.columns([1, 3])
    with col1:
        tf = st.selectbox("时间周期", ["1h", "4h", "1d"], index=0)
        # 找该 TF 现存最早/最晚日期
        files = sorted(DATA_DIR.glob(f"{tf}_*.parquet"))
        if not files:
            st.error(f"无 {tf} 数据"); return
        df_min = pl.read_parquet(files[0])["timestamp"].min()
        df_max = pl.read_parquet(files[-1])["timestamp"].max()
        # 默认显示最近 90 天
        d_default_end = df_max.date()
        d_default_start = (df_max - timedelta(days=90)).date()
        d_start = st.date_input("起始日期", d_default_start,
                                min_value=df_min.date(), max_value=df_max.date())
        d_end = st.date_input("结束日期", d_default_end,
                              min_value=df_min.date(), max_value=df_max.date())
        st.divider()
        st.markdown("**叠加技术指标**")
        show_sma = st.checkbox("SMA(20)", value=True)
        show_ema = st.checkbox("EMA(50)", value=False)
        show_bb = st.checkbox("Bollinger Bands(20, 2)", value=True)
        show_rsi = st.checkbox("RSI(14) (副图)", value=True)
        show_macd = st.checkbox("MACD(12,26,9) (副图)", value=False)
        st.divider()
        st.markdown("**市场情绪**")
        show_fgi = st.checkbox("贪婪恐慌指数", value=True)
        show_ls = st.checkbox("散户多空比", value=False)
        show_oi = st.checkbox("持仓量 OI", value=False)

    # 加载并切片
    raw = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").unique(subset=["timestamp"]).sort("timestamp")
    ts_start = datetime(d_start.year, d_start.month, d_start.day, tzinfo=timezone.utc)
    ts_end = datetime(d_end.year, d_end.month, d_end.day, 23, 59, 59, tzinfo=timezone.utc)
    sliced = raw.filter((pl.col("timestamp") >= ts_start) & (pl.col("timestamp") <= ts_end))
    if sliced.height == 0:
        st.warning("时段内无数据")
        return

    # 指标配置
    ind_list: list[tuple[str, dict]] = []
    if show_sma: ind_list.append(("sma", {"period": 20}))
    if show_ema: ind_list.append(("ema", {"period": 50}))
    if show_bb: ind_list.append(("bollinger", {"period": 20, "std_dev": 2.0}))
    if show_rsi: ind_list.append(("rsi", {"period": 14}))
    if show_macd: ind_list.append(("macd", {"fast": 12, "slow": 26, "signal": 9}))
    if ind_list:
        sliced = IndicatorEngine(sliced).compute_all(ind_list)

    # 主图 K 线
    n_sub = 1 + int(show_rsi) + int(show_macd) + int(show_fgi or show_ls or show_oi)
    row_heights = [0.55] + [0.15] * (n_sub - 1)
    fig = make_subplots(rows=n_sub, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=row_heights)

    fig.add_trace(go.Candlestick(
        x=sliced["timestamp"].to_list(),
        open=sliced["open"].to_list(), high=sliced["high"].to_list(),
        low=sliced["low"].to_list(), close=sliced["close"].to_list(),
        name="BTC", increasing_line_color="#3fb950", decreasing_line_color="#f85149",
    ), row=1, col=1)

    if show_sma and "sma_20" in sliced.columns:
        fig.add_trace(go.Scatter(x=sliced["timestamp"], y=sliced["sma_20"],
                                 mode="lines", name="SMA(20)",
                                 line=dict(color="#f0a868", width=1.2)), row=1, col=1)
    if show_ema and "ema_50" in sliced.columns:
        fig.add_trace(go.Scatter(x=sliced["timestamp"], y=sliced["ema_50"],
                                 mode="lines", name="EMA(50)",
                                 line=dict(color="#a371f7", width=1.2)), row=1, col=1)
    if show_bb and "bb_upper_20_2.0" in sliced.columns:
        for col, color in [("bb_upper_20_2.0", "#9ba1a8"), ("bb_middle_20_2.0", "#ffd33d"),
                           ("bb_lower_20_2.0", "#9ba1a8")]:
            fig.add_trace(go.Scatter(x=sliced["timestamp"], y=sliced[col],
                                     mode="lines", name=col,
                                     line=dict(color=color, width=0.9, dash="dot")), row=1, col=1)

    cur_row = 2
    if show_rsi and "rsi_14" in sliced.columns:
        fig.add_trace(go.Scatter(x=sliced["timestamp"], y=sliced["rsi_14"],
                                 mode="lines", name="RSI(14)",
                                 line=dict(color="#4cd2ff", width=1.0)), row=cur_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#9ba1a8", row=cur_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#9ba1a8", row=cur_row, col=1)
        fig.update_yaxes(range=[0, 100], row=cur_row, col=1)
        cur_row += 1
    if show_macd and "macd_line_12_26_9" in sliced.columns:
        fig.add_trace(go.Scatter(x=sliced["timestamp"], y=sliced["macd_line_12_26_9"],
                                 mode="lines", name="MACD"), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=sliced["timestamp"], y=sliced["macd_signal_12_26_9"],
                                 mode="lines", name="Signal"), row=cur_row, col=1)
        fig.add_trace(go.Bar(x=sliced["timestamp"], y=sliced["macd_histogram_12_26_9"],
                             name="Hist", marker_color="#4cd2ff", opacity=0.5), row=cur_row, col=1)
        cur_row += 1

    # 市场情绪副图
    if show_fgi or show_ls or show_oi:
        if show_fgi and aux.get("fgi") is not None:
            fgi = aux["fgi"].filter((pl.col("timestamp") >= ts_start) & (pl.col("timestamp") <= ts_end))
            if fgi.height > 0:
                fig.add_trace(go.Scatter(x=fgi["timestamp"], y=fgi["value"],
                                         mode="lines", name="FGI",
                                         line=dict(color="#fbb500", width=1.2)), row=cur_row, col=1)
        if show_ls and aux.get("ls") is not None:
            ls = aux["ls"].filter((pl.col("timestamp") >= ts_start) & (pl.col("timestamp") <= ts_end))
            if ls.height > 0:
                # downsample to hourly
                ls_d = ls.group_by_dynamic("timestamp", every="1h").agg(pl.col("long_short_ratio").mean())
                fig.add_trace(go.Scatter(x=ls_d["timestamp"], y=ls_d["long_short_ratio"],
                                         mode="lines", name="散户多空比",
                                         line=dict(color="#a371f7", width=1.0)), row=cur_row, col=1)
        if show_oi and aux.get("oi") is not None:
            oi = aux["oi"].filter((pl.col("timestamp") >= ts_start) & (pl.col("timestamp") <= ts_end))
            if oi.height > 0:
                oi_d = oi.group_by_dynamic("timestamp", every="1h").agg(pl.col("open_interest").mean())
                fig.add_trace(go.Scatter(x=oi_d["timestamp"], y=oi_d["open_interest"],
                                         mode="lines", name="OI",
                                         line=dict(color="#f0a868", width=1.0), yaxis="y2"), row=cur_row, col=1)

    fig.update_layout(template="plotly_dark", height=160 * (n_sub) + 200,
                      showlegend=True, hovermode="x unified", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------- 页面 4：风险分析 ----------
def page_risk() -> None:
    st.title("⚠️ 风险分析")
    st.caption("蒙特卡洛 bootstrap + 杠杆扫描结果展示")

    presets = list_preset_strategies()
    choice = st.selectbox("策略", list(presets.keys()),
                          index=list(presets.keys()).index("strategies_v2_optimized")
                          if "strategies_v2_optimized" in presets else 0)
    n_sim = st.slider("模拟次数", 100, 5000, 1000, 100)
    ruin_pct = st.slider("破产阈值（× 初始资金）", 0.05, 0.50, 0.20, 0.05)

    if not st.button("🎲 跑蒙特卡洛", type="primary"):
        st.info("👆 配置后点击运行")
        scan_csv = PROJECT_ROOT / "output" / "leverage_scan" / "table.csv"
        if scan_csv.exists():
            st.subheader("📉 杠杆扫描结果（来自 leverage_scan.py）")
            st.dataframe(pl.read_csv(scan_csv).to_dicts(), use_container_width=True, hide_index=True)
        return

    text = presets[choice].read_text(encoding="utf-8")
    d = yaml.safe_load(text)
    bt_cfg = load_bt_cfg()
    used_tfs = used_timeframes(d["strategies"], bt_cfg["primary_timeframe"])
    ind_cfg = _collect_required_indicators(d["strategies"])
    aux = get_aux()
    bt_params = (
        float(bt_cfg["initial_balance"]),
        float(bt_cfg["leverage"]),
        float(bt_cfg["fee_rate"]),
        float(bt_cfg["slippage"]),
        float(bt_cfg["maintenance_margin_rate"]),
        float(bt_cfg["max_drawdown_pct"]),
        bt_cfg["primary_timeframe"],
        bt_cfg.get("daily_max_loss_pct"),
    )
    result = run_backtest_cached(
        yaml_text=text, bt_params=bt_params,
        used_tfs_tuple=tuple(sorted(used_tfs)),
        ind_cfg_key=_ind_cfg_to_key(ind_cfg),
        has_funding=aux.get("funding") is not None,
    )

    n_closed = sum(1 for t in result.trades
                   if t.side.endswith("_close") or t.side == "liquidate")
    if n_closed < 5:
        st.error("已平仓交易过少，无法 MC")
        return

    mc = MonteCarloSimulator(result)
    mcr = mc.run(n_simulations=n_sim, ruin_fraction=ruin_pct, seed=42)

    cols = st.columns(4)
    cols[0].metric("破产概率", f"{mcr.ruin_probability * 100:.2f}%")
    cols[1].metric("收益中位数", f"{mcr.percentiles(mcr.final_returns_pct)[50]:+.1f}%")
    cols[2].metric("收益 P5", f"{mcr.percentiles(mcr.final_returns_pct)[5]:+.1f}%")
    cols[3].metric("回撤 P95", f"{mcr.percentiles(mcr.max_drawdowns_pct)[95]:.1f}%")

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12, row_heights=[0.5, 0.5],
                        subplot_titles=("最终收益率分布 (%)", "最大回撤分布 (%)"))
    fig.add_trace(go.Histogram(x=mcr.final_returns_pct, nbinsx=60, name="收益",
                               marker_color="#3fb950"), row=1, col=1)
    fig.add_trace(go.Histogram(x=mcr.max_drawdowns_pct, nbinsx=60, name="回撤",
                               marker_color="#ff6b6b"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, showlegend=False,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 杠杆扫描参考
    scan_csv = PROJECT_ROOT / "output" / "leverage_scan" / "table.csv"
    if scan_csv.exists():
        st.subheader("📉 杠杆扫描结果（来自 leverage_scan.py）")
        st.dataframe(pl.read_csv(scan_csv).to_dicts(), use_container_width=True, hide_index=True)


# ---------- 入口 ----------
def main() -> None:
    st.set_page_config(
        page_title="quant-btc 仪表盘",
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("📈 quant-btc")
    st.sidebar.caption("BTC/USDT 永续合约符号主义量化系统")

    page = st.sidebar.radio(
        "页面",
        ["策略回测", "策略对比", "数据浏览", "风险分析"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    if page == "策略回测":
        page_backtest()
    elif page == "策略对比":
        page_compare()
    elif page == "数据浏览":
        page_data()
    elif page == "风险分析":
        page_risk()


if __name__ == "__main__":
    main()
