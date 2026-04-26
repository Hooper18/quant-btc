"""参数网格优化器。

工作流：
1. 加载 base 策略 YAML 作为模板
2. itertools.product 生成所有参数组合
3. 数据按时间分为训练集（前 train_ratio）/ 测试集（后 1-train_ratio）
4. 每个组合：修改模板参数 → 训练集回测 → 记录指标
5. 按指定 metric 排序找最优
6. 最优参数在测试集再跑一次
7. 过拟合检测：test_sharpe < 0.5 × train_sharpe（且 train_sharpe > 0）→ 警告

数据 + 指标必须在调用 optimize 前就准备好；optimizer 不重算指标。
"""
from __future__ import annotations

import copy
import itertools
import logging
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from .backtester import Backtester, BacktestResult

logger = logging.getLogger(__name__)


# ---------- 路径解析：'strategies[0].conditions[0].value' → 写入器 ----------
_PATH_TOKEN_RE = re.compile(r"^([^\[\]]+)((?:\[\d+\])*)$")
_BRACKET_IDX_RE = re.compile(r"\[(\d+)\]")


def _parse_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in path.split("."):
        m = _PATH_TOKEN_RE.match(part)
        if not m:
            raise ValueError(f"无法解析路径片段：{part!r}")
        tokens.append(m.group(1))
        for idx_match in _BRACKET_IDX_RE.finditer(m.group(2)):
            tokens.append(int(idx_match.group(1)))
    return tokens


def set_param(obj: Any, path: str, value: Any) -> None:
    """按 dotted-path 写入嵌套结构（dict / list 混合）。"""
    tokens = _parse_path(path)
    parent = obj
    for t in tokens[:-1]:
        parent = parent[t]
    parent[tokens[-1]] = value


def get_param(obj: Any, path: str) -> Any:
    tokens = _parse_path(path)
    cur = obj
    for t in tokens:
        cur = cur[t]
    return cur


# ---------- metric 字典 ----------
_METRIC_KEYS = {
    "sharpe": "sharpe_ratio",
    "return": "total_return_pct",
    "annual_return": "annualized_return_pct",
    "max_drawdown": "max_drawdown_pct",
    "profit_loss_ratio": "profit_loss_ratio",
    "win_rate": "win_rate_pct",
}
_LARGER_IS_BETTER = {
    "sharpe": True, "return": True, "annual_return": True,
    "max_drawdown": False, "profit_loss_ratio": True, "win_rate": True,
}


@dataclass
class OptimizeResult:
    rows: list[dict[str, Any]]      # 每个组合：参数 + 训练集指标
    best_params: dict[str, Any]
    best_train: BacktestResult
    best_test: BacktestResult
    overfit: bool
    metric: str


class StrategyOptimizer:
    def __init__(
        self,
        base_strategy_path: str | Path,
        data_dict: dict[str, pl.DataFrame],
        backtest_config_path: str | Path = "config/backtest_config.yaml",
    ):
        self.base_path = Path(base_strategy_path)
        with self.base_path.open(encoding="utf-8") as f:
            self.base_yaml: dict[str, Any] = yaml.safe_load(f) or {}
        self.data_dict = data_dict
        self.backtester = Backtester.from_yaml(backtest_config_path)
        if self.backtester.primary_tf not in data_dict:
            raise ValueError(
                f"data_dict 缺主 TF={self.backtester.primary_tf}；已有 {list(data_dict)}"
            )

    # ---------- 数据切分 ----------
    def _split(self, train_ratio: float) -> tuple[dict, dict, Any]:
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio 必须在 (0,1)，收到 {train_ratio}")
        primary = self.data_dict[self.backtester.primary_tf]
        n = primary.height
        split_idx = int(n * train_ratio)
        if split_idx <= 0 or split_idx >= n:
            raise ValueError(f"train_ratio={train_ratio} 切出空集（n={n}）")
        split_ts = primary["timestamp"][split_idx]
        train = {tf: df.filter(pl.col("timestamp") < split_ts) for tf, df in self.data_dict.items()}
        test = {tf: df.filter(pl.col("timestamp") >= split_ts) for tf, df in self.data_dict.items()}
        return train, test, split_ts

    # ---------- YAML 写入与回测 ----------
    def make_strategy_yaml(self, params: dict[str, Any]) -> dict[str, Any]:
        y = copy.deepcopy(self.base_yaml)
        for path, value in params.items():
            set_param(y, path, value)
        return y

    def _run(
        self,
        strategy_yaml: dict[str, Any],
        data_dict: dict[str, pl.DataFrame],
        fr_df: pl.DataFrame | None,
    ) -> BacktestResult:
        # tempfile.NamedTemporaryFile 在 Windows 下被句柄占用难处理；用 uuid 命名
        tmp_path = Path(tempfile.gettempdir()) / f"opt_strat_{uuid.uuid4().hex}.yaml"
        tmp_path.write_text(
            yaml.safe_dump(strategy_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        try:
            return self.backtester.run(data_dict, str(tmp_path), funding_rate_df=fr_df)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ---------- 主入口 ----------
    def optimize(
        self,
        param_grid: dict[str, list[Any]],
        metric: str = "sharpe",
        train_ratio: float = 0.7,
        funding_rate_df: pl.DataFrame | None = None,
        progress: bool = True,
    ) -> OptimizeResult:
        if metric not in _METRIC_KEYS:
            raise ValueError(f"未知 metric={metric}；可选 {list(_METRIC_KEYS)}")
        metric_key = _METRIC_KEYS[metric]
        larger_better = _LARGER_IS_BETTER[metric]

        train_data, test_data, split_ts = self._split(train_ratio)
        primary_tf = self.backtester.primary_tf
        logger.info(
            "数据切分 @ %s（训练集 %d bars / 测试集 %d bars）",
            split_ts, train_data[primary_tf].height, test_data[primary_tf].height,
        )

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combos = list(itertools.product(*values))
        total = len(combos)
        logger.info("参数组合数：%d（维度=%d）", total, len(keys))

        rows: list[dict[str, Any]] = []
        best_score: float | None = None
        best_idx: int = -1

        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            try:
                strat_yaml = self.make_strategy_yaml(params)
                res = self._run(strat_yaml, train_data, funding_rate_df)
            except Exception as e:
                logger.exception("[%d/%d] 失败：%s", i, total, params)
                if progress:
                    print(f"[{i:3d}/{total}] FAILED: {e!s:.80s}")
                continue

            score = float(res.metrics.get(metric_key, 0))
            row = {
                **params,
                "train_total_return_pct": float(res.metrics.get("total_return_pct", 0)),
                "train_sharpe": float(res.metrics.get("sharpe_ratio", 0)),
                "train_max_dd_pct": float(res.metrics.get("max_drawdown_pct", 0)),
                "train_win_rate_pct": float(res.metrics.get("win_rate_pct", 0)),
                "train_pl_ratio": float(res.metrics.get("profit_loss_ratio", 0)),
                "train_total_trades": int(res.metrics.get("total_trades", 0)),
                "train_circuit_breaker": int(res.metrics.get("circuit_breaker", 0)),
            }
            rows.append(row)

            improved = (
                best_score is None
                or (larger_better and score > best_score)
                or (not larger_better and score < best_score)
            )
            if improved:
                best_score = score
                best_idx = len(rows) - 1
                marker = " ★"
            else:
                marker = ""
            if progress:
                print(
                    f"[{i:3d}/{total}] {metric}={score:+8.3f}  "
                    f"best={best_score:+8.3f}{marker}  {params}"
                )

        if best_idx < 0:
            raise RuntimeError("所有参数组合都回测失败")

        # 在测试集上验证最优参数
        best_row = rows[best_idx]
        best_params = {k: best_row[k] for k in keys}
        logger.info("最优参数：%s（train %s=%.4f）", best_params, metric, best_score)
        best_train = self._run(self.make_strategy_yaml(best_params), train_data, funding_rate_df)
        best_test = self._run(self.make_strategy_yaml(best_params), test_data, funding_rate_df)

        # 过拟合检测：仅在 train sharpe > 0 时才有意义
        train_sharpe = float(best_train.metrics.get("sharpe_ratio", 0))
        test_sharpe = float(best_test.metrics.get("sharpe_ratio", 0))
        overfit = train_sharpe > 0 and test_sharpe < 0.5 * train_sharpe

        return OptimizeResult(
            rows=rows,
            best_params=best_params,
            best_train=best_train,
            best_test=best_test,
            overfit=overfit,
            metric=metric,
        )
