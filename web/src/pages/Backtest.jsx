import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import EquityChart from '../components/EquityChart'
import MonthlyHeatmap from '../components/MonthlyHeatmap'
import TradeTable from '../components/TradeTable'
import MetricCard from '../components/MetricCard'
import { fmtPct, fmtNum } from '../utils/dataLoader'
import { loadStrategiesIndex, loadStrategyDetail } from '../utils/strategyData'

const SYMBOL_OPTIONS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

export default function Backtest() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [index, setIndex] = useState(null)
  const [detail, setDetail] = useState(null)
  const [strategyId, setStrategyId] = useState(null)
  const [symbol, setSymbol] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadStrategiesIndex()
      .then((idx) => {
        setIndex(idx)
        const fromUrl = searchParams.get('strategy')
        const exists = idx.strategies.find((s) => s.id === fromUrl)
        const initId = exists ? fromUrl : idx.strategies[0]?.id
        setStrategyId(initId || null)
      })
      .catch((err) => setError(err.message))
  }, [])

  useEffect(() => {
    if (!strategyId) return
    setDetail(null)
    loadStrategyDetail(strategyId)
      .then((d) => {
        setDetail(d)
        const sym = searchParams.get('symbol')
        const fallback = d.default_symbol || d.applicable_symbols[0]
        setSymbol(d.applicable_symbols.includes(sym) ? sym : fallback)
      })
      .catch((err) => setError(err.message))
  }, [strategyId])

  const onSelectStrategy = (id) => {
    setStrategyId(id)
    setSearchParams({ strategy: id })
  }

  const blob = symbol && detail ? detail.by_symbol[symbol] : null
  const metrics = blob?.metrics

  const indexEntry = useMemo(
    () => index?.strategies.find((s) => s.id === strategyId) || null,
    [index, strategyId],
  )

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <h1 className="text-xl font-semibold">回测详情</h1>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs text-gray-500">策略</label>
          <select
            value={strategyId || ''}
            onChange={(e) => onSelectStrategy(e.target.value)}
            className="bg-gray-900 border border-gray-800 rounded-md px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-sky-600"
          >
            {(index?.strategies || []).map((s) => (
              <option key={s.id} value={s.id}>
                {s.display_name}
              </option>
            ))}
          </select>
          {detail && detail.applicable_symbols.length > 1 && (
            <div className="flex gap-1 rounded-md bg-gray-900 border border-gray-800 p-1">
              {SYMBOL_OPTIONS.filter((s) => detail.applicable_symbols.includes(s)).map((s) => (
                <button
                  key={s}
                  onClick={() => setSymbol(s)}
                  className={`px-3 py-1 rounded text-xs ${
                    symbol === s ? 'bg-sky-600 text-white' : 'text-gray-400 hover:text-gray-100'
                  }`}
                >
                  {s.replace('USDT', '')}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded bg-rose-950/40 border border-rose-900 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      {detail && (
        <section className="rounded-lg bg-gray-900 border border-gray-800 p-5 space-y-3">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2 className="text-2xl font-semibold text-gray-100">{detail.display_name}</h2>
            <span className="text-xs text-gray-500">
              {detail.category} · 配置文件 <code className="bg-gray-950 px-1.5 py-0.5 rounded">{detail.file}</code>
            </span>
          </div>
          {detail.tags?.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {detail.tags.map((t) => (
                <span
                  key={t}
                  className="px-2 py-0.5 rounded text-[11px] bg-sky-950/60 text-sky-300 border border-sky-900/60"
                >
                  {t}
                </span>
              ))}
            </div>
          )}
          <div className="text-sm text-gray-400">
            <span className="text-gray-500">适用币种：</span>
            {detail.applicable_symbols.map((s) => s.replace('USDT', '')).join(' / ')}
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">策略规则</div>
            <ol className="space-y-1.5 text-sm text-gray-300 leading-relaxed">
              {detail.rule_items.map((it, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-gray-600 tabular-nums">{i + 1}.</span>
                  <span>
                    <span className="text-gray-100 font-medium">{it.name}</span>
                    <span className="text-gray-500"> · </span>
                    <span>当 </span>
                    <span className="text-amber-200">{it.conditions_text}</span>
                    <span> → </span>
                    <span className={it.side === 'long' ? 'text-emerald-300' : 'text-rose-300'}>
                      {it.side_zh}
                    </span>
                    <span className="text-gray-500">
                      （仓位 {it.size_pct}% · 止损 {it.stop_loss_pct}% · 止盈 {it.take_profit_pct}%）
                    </span>
                  </span>
                </li>
              ))}
            </ol>
          </div>
          {indexEntry?.composite_score != null && (
            <div className="text-xs text-gray-500">
              综合评分 <span className="text-sky-300 tabular-nums">{indexEntry.composite_score.toFixed(2)}</span>
              <span className="ml-2">/ 满分 100（夏普 0.3 + 年化 0.3 + 抗回撤 0.2 + 胜率 0.2）</span>
            </div>
          )}
        </section>
      )}

      <section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="总收益率" tone="up" value={fmtPct(metrics?.total_return_pct ?? null, 1)} />
        <MetricCard label="年化" tone="up" value={fmtPct(metrics?.annualized_return_pct ?? null, 1)} />
        <MetricCard label="夏普" tone="accent" value={fmtNum(metrics?.sharpe_ratio, 2)} />
        <MetricCard label="最大回撤" tone="down" value={fmtPct(metrics?.max_drawdown_pct ?? null, 1)} />
        <MetricCard label="胜率" value={fmtPct(metrics?.win_rate_pct ?? null, 1)} />
        <MetricCard
          label="交易笔数"
          value={fmtNum(metrics?.total_trades, 0)}
          sub={`平均持仓 ${fmtNum(metrics?.avg_holding_hours, 1)} h · 盈亏比 ${fmtNum(metrics?.profit_loss_ratio, 2)}`}
        />
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">
          净值曲线（{symbol?.replace('USDT', '') || '—'}）
        </h2>
        <EquityChart points={blob?.equity_points || []} height={340} />
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">月度收益率热力图</h2>
        <MonthlyHeatmap matrix={blob?.monthly_matrix || []} />
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-gray-400">交易记录（最近 100 笔）</h2>
          <span className="text-xs text-gray-500">点表头排序</span>
        </div>
        <TradeTable rows={[...(blob?.trades?.rows || [])].reverse()} pageSize={20} />
      </section>
    </div>
  )
}
