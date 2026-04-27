import { useEffect, useState } from 'react'
import EquityChart from '../components/EquityChart'
import MonthlyHeatmap from '../components/MonthlyHeatmap'
import TradeTable from '../components/TradeTable'
import MetricCard from '../components/MetricCard'
import { loadJson, fmtPct, fmtNum } from '../utils/dataLoader'

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

export default function Backtest() {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [bundle, setBundle] = useState(null)
  const [wf, setWf] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setBundle(null)
    setError(null)
    Promise.all([
      loadJson(`${symbol}_metrics.json`),
      loadJson(`${symbol}_equity_curve.json`),
      loadJson(`${symbol}_monthly_returns.json`),
      loadJson(`${symbol}_trades.json`),
    ])
      .then(([m, e, mr, tr]) => {
        setBundle({ metrics: m, equity: e, monthly: mr, trades: tr })
      })
      .catch((err) => setError(err.message))
  }, [symbol])

  useEffect(() => {
    loadJson('walk_forward_summary.json').then(setWf).catch(() => {})
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">回测详情</h1>
        <div className="flex gap-1 rounded-md bg-gray-900 border border-gray-800 p-1">
          {SYMBOLS.map((s) => (
            <button
              key={s}
              onClick={() => setSymbol(s)}
              className={`px-3 py-1 rounded text-xs ${
                symbol === s
                  ? 'bg-sky-600 text-white'
                  : 'text-gray-400 hover:text-gray-100'
              }`}
            >
              {s.replace('USDT', '')}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded bg-rose-950/40 border border-rose-900 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="总收益率" tone="up"
          value={fmtPct(bundle?.metrics?.total_return_pct ?? null, 1)} />
        <MetricCard label="年化"
          value={fmtPct(bundle?.metrics?.annualized_return_pct ?? null, 1)} tone="up" />
        <MetricCard label="夏普" tone="accent"
          value={fmtNum(bundle?.metrics?.sharpe_ratio, 2)} />
        <MetricCard label="最大回撤" tone="down"
          value={fmtPct(bundle?.metrics?.max_drawdown_pct ?? null, 1)} />
        <MetricCard label="胜率"
          value={fmtPct(bundle?.metrics?.win_rate_pct ?? null, 1)} />
        <MetricCard label="交易笔数"
          value={fmtNum(bundle?.metrics?.total_trades, 0)}
          sub={`平均持仓 ${fmtNum(bundle?.metrics?.avg_holding_hours, 1)} h`} />
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">净值曲线</h2>
        <EquityChart points={bundle?.equity?.points || []} height={340} />
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">月度收益率热力图</h2>
        <MonthlyHeatmap matrix={bundle?.monthly?.matrix || []} />
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-gray-400">
            Walk-Forward 20 窗口（汇总参数对策略）
          </h2>
          {wf?.summary && (
            <span className="text-xs text-gray-500">
              汇总：年化 {wf.summary.annualized_return_pct?.toFixed(1)}% · 夏普{' '}
              {wf.summary.sharpe_ratio?.toFixed(2)} · 回撤{' '}
              {wf.summary.max_drawdown_pct?.toFixed(1)}%
            </span>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr>
                {['#', '训练期', '测试期', '训练收益%', '训练夏普', '测试收益%', '测试夏普', '测试回撤%', '测试胜率%', '交易'].map((h) => (
                  <th key={h} className="px-2 py-1.5 text-left font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(wf?.windows || []).map((w) => (
                <tr key={w.window} className="border-b border-gray-900/60 hover:bg-gray-900/40">
                  <td className="px-2 py-1.5 text-gray-400">{w.window}</td>
                  <td className="px-2 py-1.5 text-gray-400 tabular-nums">
                    {w.train_start.slice(0, 10)} → {w.train_end.slice(0, 10)}
                  </td>
                  <td className="px-2 py-1.5 text-gray-400 tabular-nums">
                    {w.test_start.slice(0, 10)} → {w.test_end.slice(0, 10)}
                  </td>
                  <td className="px-2 py-1.5 text-gray-200 tabular-nums">{w.train_return_pct.toFixed(1)}</td>
                  <td className="px-2 py-1.5 text-gray-200 tabular-nums">{w.train_sharpe.toFixed(2)}</td>
                  <td className={`px-2 py-1.5 tabular-nums ${w.test_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {w.test_return_pct.toFixed(1)}
                  </td>
                  <td className={`px-2 py-1.5 tabular-nums ${w.test_sharpe >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {w.test_sharpe.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5 text-amber-300 tabular-nums">{w.test_max_dd_pct.toFixed(1)}</td>
                  <td className="px-2 py-1.5 text-gray-300 tabular-nums">{w.test_win_rate_pct.toFixed(1)}</td>
                  <td className="px-2 py-1.5 text-gray-400 tabular-nums">{w.test_total_trades}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-gray-400">交易记录（最近 100 笔）</h2>
          <span className="text-xs text-gray-500">点表头排序</span>
        </div>
        <TradeTable rows={[...(bundle?.trades?.rows || [])].reverse()} pageSize={20} />
      </section>
    </div>
  )
}
