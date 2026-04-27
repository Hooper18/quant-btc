import { useEffect, useState } from 'react'
import LivePrice from '../components/LivePrice'
import MetricCard from '../components/MetricCard'
import EquityChart from '../components/EquityChart'
import TradeTable from '../components/TradeTable'
import { loadJson, fmtPct, fmtNum } from '../utils/dataLoader'

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [equity, setEquity] = useState(null)
  const [trades, setTrades] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      loadJson('BTCUSDT_metrics.json'),
      loadJson('BTCUSDT_equity_curve.json'),
      loadJson('BTCUSDT_trades.json'),
    ])
      .then(([m, e, t]) => {
        setMetrics(m)
        setEquity(e)
        setTrades(t)
      })
      .catch((err) => setError(err.message))
  }, [])

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-xl font-semibold text-gray-100">实时行情</h1>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
          <LivePrice symbol="BTCUSDT" />
          <LivePrice symbol="ETHUSDT" />
          <LivePrice symbol="SOLUSDT" />
        </div>
      </section>

      {error && (
        <div className="rounded bg-rose-950/40 border border-rose-900 px-4 py-3 text-sm text-rose-300">
          数据加载失败：{error}
        </div>
      )}

      <section>
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">
          BTC v2 最优策略 · 关键指标
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          <MetricCard
            label="总收益率"
            value={fmtPct(metrics?.total_return_pct ?? null, 1)}
            tone="up"
            sub={`期末 ${fmtNum(metrics?.final_equity, 2)} USDT`}
          />
          <MetricCard
            label="年化收益率"
            value={fmtPct(metrics?.annualized_return_pct ?? null, 1)}
            tone="up"
          />
          <MetricCard
            label="夏普比率"
            value={fmtNum(metrics?.sharpe_ratio, 2)}
            tone="accent"
          />
          <MetricCard
            label="最大回撤"
            value={fmtPct(metrics?.max_drawdown_pct ?? null, 1)}
            tone="down"
            sub={`熔断 ${metrics?.circuit_breaker ? '触发' : '未触发'}`}
          />
          <MetricCard
            label="胜率"
            value={fmtPct(metrics?.win_rate_pct ?? null, 1)}
            sub={`${metrics?.total_trades ?? '—'} 笔 · 盈亏比 ${fmtNum(metrics?.profit_loss_ratio, 2)}`}
          />
        </div>
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 rounded-lg bg-gray-900 border border-gray-800 p-4">
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-sm uppercase tracking-wider text-gray-400">
              净值曲线（左轴）vs BTC 价格（右轴）
            </h2>
            <span className="text-xs text-gray-500">日采样</span>
          </div>
          <EquityChart points={equity?.points || []} height={340} />
        </div>
        <div className="rounded-lg bg-gray-900 border border-gray-800 p-4">
          <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">
            最近 10 笔交易
          </h2>
          <TradeTable rows={(trades?.rows || []).slice(-10).reverse()} pageSize={10} compact />
        </div>
      </section>
    </div>
  )
}
