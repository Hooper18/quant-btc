"""参数敏感度热力图。

固定 best_strategy 的其余参数，每次只变两个，扫一张二维网格 → 夏普热力图。
当前最优参数在图上用 ★ 标注；如果它落在网格边缘（首/末行 or 首/末列），打印警告
建议扩大搜索范围。

输出：output/sensitivity/<pair_name>.png
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest import Backtester  # noqa: E402
from backtest.optimizer import set_param  # noqa: E402
from backtest.visualizer import _DARK_RC  # noqa: E402
from utils.config import DataConfig  # noqa: E402

# 复用 run_backtest 的数据加载 + 指标解析
from run_backtest import (  # noqa: E402
    _collect_required_indicators,
    build_data_dict,
    load_aux_data,
)


# best_strategy 的当前最优参数（Phase 9 grid search 结果）
BEST_PARAMS = {
    "strategies[0].conditions[0].value": 70,    # RSI 阈值
    "strategies[0].action.size_pct": 5,          # 仓位
    "strategies[0].stop_loss_pct": 5,            # 止损
    "strategies[0].take_profit_pct": 6,          # 止盈
}

# 4 个待扫的参数对：(p1_path, p1_label, p1_values, p2_path, p2_label, p2_values, file_name)
HEATMAPS = [
    (
        "strategies[0].conditions[0].value", "RSI阈值", [60, 65, 70, 75, 80, 85, 90],
        "strategies[0].action.size_pct",     "仓位%",    [3, 5, 7, 10, 15],
        "rsi_x_size",
    ),
    (
        "strategies[0].conditions[0].value", "RSI阈值", [60, 65, 70, 75, 80, 85, 90],
        "strategies[0].stop_loss_pct",       "止损%",    [1, 2, 3, 4, 5, 7],
        "rsi_x_sl",
    ),
    (
        "strategies[0].conditions[0].value", "RSI阈值", [60, 65, 70, 75, 80, 85, 90],
        "strategies[0].take_profit_pct",     "止盈%",    [3, 4, 5, 6, 8, 10],
        "rsi_x_tp",
    ),
    (
        "strategies[0].stop_loss_pct",       "止损%",    [1, 2, 3, 4, 5, 7],
        "strategies[0].take_profit_pct",     "止盈%",    [3, 4, 5, 6, 8, 10],
        "sl_x_tp",
    ),
]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("sensitivity").setLevel(logging.INFO)


def _used_timeframes(strategies: list[dict], primary: str) -> set[str]:
    used: set[str] = {primary}
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


def _run_one(
    strat_yaml: dict, data_dict: dict[str, pl.DataFrame],
    bt: Backtester, fr_df: pl.DataFrame | None,
) -> float:
    """跑一次回测，返回 sharpe。失败返回 nan。"""
    tmp_path = Path(tempfile.gettempdir()) / f"sens_{uuid.uuid4().hex}.yaml"
    tmp_path.write_text(
        yaml.safe_dump(strat_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    try:
        res = bt.run(data_dict, str(tmp_path), funding_rate_df=fr_df)
        return float(res.metrics.get("sharpe_ratio", 0))
    except Exception:
        return float("nan")
    finally:
        tmp_path.unlink(missing_ok=True)


def _scan_grid(
    base_yaml: dict,
    p1_path: str, p1_values: list[Any],
    p2_path: str, p2_values: list[Any],
    bt: Backtester,
    data_dict: dict[str, pl.DataFrame],
    fr_df: pl.DataFrame | None,
    label: str,
) -> np.ndarray:
    """二维扫描，返回 shape=(len(p1), len(p2)) 的 sharpe 矩阵。"""
    matrix = np.full((len(p1_values), len(p2_values)), np.nan)
    total = len(p1_values) * len(p2_values)
    cnt = 0
    for i, v1 in enumerate(p1_values):
        for j, v2 in enumerate(p2_values):
            mod = copy.deepcopy(base_yaml)
            # 先把所有 best 参数应用到模板，再覆盖待扫两个
            for path, val in BEST_PARAMS.items():
                set_param(mod, path, val)
            set_param(mod, p1_path, v1)
            set_param(mod, p2_path, v2)
            sharpe = _run_one(mod, data_dict, bt, fr_df)
            matrix[i, j] = sharpe
            cnt += 1
            print(f"  [{label}] [{cnt:3d}/{total}] {p1_path.split('.')[-1]}={v1} "
                  f"{p2_path.split('.')[-1]}={v2} → sharpe={sharpe:+.3f}")
    return matrix


def _plot_heatmap(
    matrix: np.ndarray,
    p1_label: str, p1_values: list[Any],
    p2_label: str, p2_values: list[Any],
    best_p1: Any, best_p2: Any,
    out_path: Path,
) -> bool:
    """绘制热力图，最优参数处标 ★。返回是否在边缘。"""
    plt.rcParams.update(_DARK_RC)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(p2_values) + 4),
                                    max(6, 0.7 * len(p1_values) + 3)))
    bound = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1e-6)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(p2_values)))
    ax.set_xticklabels([str(v) for v in p2_values])
    ax.set_yticks(range(len(p1_values)))
    ax.set_yticklabels([str(v) for v in p1_values])
    ax.set_xlabel(p2_label)
    ax.set_ylabel(p1_label)
    ax.set_title(f"敏感度：{p1_label} × {p2_label} → 夏普")

    # 标注每格数值
    for i in range(len(p1_values)):
        for j in range(len(p2_values)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(v) > bound * 0.5 else "#101010")

    # 当前最优 ★
    on_edge = False
    if best_p1 in p1_values and best_p2 in p2_values:
        bi = p1_values.index(best_p1)
        bj = p2_values.index(best_p2)
        ax.scatter([bj], [bi], marker="*", s=320,
                   color="#fefefe", edgecolor="#0e1117", linewidth=1.5, zorder=4)
        on_edge = (bi in (0, len(p1_values) - 1)) or (bj in (0, len(p2_values) - 1))

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return on_edge


def main() -> int:
    parser = argparse.ArgumentParser(description="参数敏感度热力图")
    parser.add_argument("--strategy", default=str(
        PROJECT_ROOT / "output" / "optimize_20260426_230903" / "best_strategy.yaml"
    ))
    parser.add_argument("--backtest", default=str(PROJECT_ROOT / "config" / "backtest_config.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "config" / "data_config.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "sensitivity"))
    args = parser.parse_args()

    _setup_logging()
    log = logging.getLogger("sensitivity")

    # 加载策略 YAML 模板
    with open(args.strategy, encoding="utf-8") as f:
        base_yaml = yaml.safe_load(f) or {}
    strategies = base_yaml.get("strategies", [])

    data_cfg = DataConfig.from_yaml(args.data_config)
    bt = Backtester.from_yaml(args.backtest)

    # 准备数据 + 指标
    used_tfs = _used_timeframes(strategies, bt.primary_tf)
    ind_cfg = _collect_required_indicators(strategies)
    aux = load_aux_data(data_cfg)
    log.info("加载数据：TF=%s 指标=%d 项", sorted(used_tfs), len(ind_cfg))
    data_dict = build_data_dict(data_cfg, used_tfs, ind_cfg, aux=aux)
    fr_df = aux.get("funding")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_warnings: list[str] = []
    for p1_path, p1_label, p1_values, p2_path, p2_label, p2_values, file_name in HEATMAPS:
        log.info("扫描：%s × %s（%dx%d=%d 格）",
                 p1_label, p2_label, len(p1_values), len(p2_values),
                 len(p1_values) * len(p2_values))
        matrix = _scan_grid(
            base_yaml, p1_path, p1_values, p2_path, p2_values,
            bt, data_dict, fr_df, file_name,
        )
        out_path = out_dir / f"{file_name}.png"
        on_edge = _plot_heatmap(
            matrix, p1_label, p1_values, p2_label, p2_values,
            best_p1=BEST_PARAMS[p1_path], best_p2=BEST_PARAMS[p2_path],
            out_path=out_path,
        )
        log.info("已保存 %s", out_path.name)
        if on_edge:
            edge_warnings.append(f"{p1_label} × {p2_label}")

    print("\n========== 敏感度热力图汇总 ==========")
    print(f"输出目录：{out_dir}")
    for _, p1_label, _, _, p2_label, _, fn in HEATMAPS:
        print(f"  - {fn}.png  ({p1_label} × {p2_label})")
    print(f"\n当前最优参数：{BEST_PARAMS}")
    if edge_warnings:
        for hm in edge_warnings:
            print(f"⚠️  最优参数在 {hm} 热力图边缘 → 参数可能未收敛，建议扩大搜索范围")
    else:
        print("✓  所有热力图最优参数都在内部，未触及边缘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
