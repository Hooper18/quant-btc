import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Trophy, Activity, BarChart3, ArrowRight } from 'lucide-react'
import LivePrice from '../components/LivePrice'
import MetricCard from '../components/MetricCard'
import EquityChart from '../components/EquityChart'
import Sparkline from '../components/Sparkline'
import { loadJson, fmtPct, fmtNum } from '../utils/dataLoader'
import { loadStrategiesIndex, bestStrategy, strategySummary } from '../utils/strategyData'

function fngTone(value) {
  if (value == null) return { color: '#6b7280', label: '—' }
  if (value <= 24) return { color: '#dc2626', label: '极度恐慌（潜在抄底机会）' }
  if (value <= 49) return { color: '#f59e0b', label: '恐慌（市场情绪偏空）' }
  if (value <= 54) return { color: '#facc15', label: '中性' }
  if (value <= 74) return { color: '#84cc16', label: '贪婪（注意获利了结）' }
  return { color: '#22c55e', label: '极度贪婪（警惕回调）' }
}

export default function Dashboard() {
  const [index, setIndex] = useState(null)
  const [bestEquity, setBestEquity] = useState(null)
  const [fng, setFng] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadStrategiesIndex()
      .then(setIndex)
      .catch((e) => setError(e.message))
    loadJson('fear_greed_latest.json').then(setFng).catch(() => {})
  }, [])

  const top = useMemo(() => bestStrategy(index), [index])
  const summary = useMemo(() => strategySummary(index), [index])

  useEffect(() => {
    if (!top) return
    loadJson(`strategies/${top.id}.json`)
      .then((d) => {
        const sym = d.default_symbol
        setBestEquity({
          symbol: sym,
          points: d.by_symbol[sym].equity_points,
          rules: d.rule_items,
          name: d.display_name,
        })
      })
      .catch(() => {})
  }, [top])

  const fngLast = fng?.rows?.length ? fng.rows[fng.rows.length - 1] : null
  const fngInfo = fngTone(fngLast?.value)

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

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        {/* 最佳策略 */}
        <Link
          to={top ? `/backtest?strategy=${top.id}` : '/rankings'}
          className="rounded-lg border border-amber-500/40 bg-gradient-to-br from-amber-950/30 to-gray-900 p-4 hover:border-amber-400/60 transition"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-amber-300">
              <Trophy size={14} /> 综合排名第 1 · 最佳策略
            </div>
            <ArrowRight size={14} className="text-gray-500" />
          </div>
          <div className="text-xl font-semibold text-gray-100 mt-2">
            {top?.display_name ?? '加载中…'}
          </div>
          <div className="text-[11px] text-gray-500 mt-0.5">{top?.category}</div>
          {top && (
            <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
              <div>
                <div className="text-gray-500">总收益</div>
                <div className="text-emerald-400 tabular-nums font-semibold">
                  {fmtPct(top.primary.total_return_pct, 1)}
                </div>
              </div>
              <div>
                <div className="text-gray-500">夏普</div>
                <div className="text-sky-300 tabular-nums font-semibold">
                  {fmtNum(top.primary.sharpe_ratio, 2)}
                </div>
              </div>
              <div>
                <div className="text-gray-500">回撤</div>
                <div className="text-amber-300 tabular-nums font-semibold">
                  {fmtPct(top.primary.max_drawdown_pct, 1)}
                </div>
              </div>
            </div>
          )}
          {top?.rules_text && (
            <div className="text-[11px] text-gray-400 mt-3 leading-relaxed line-clamp-3 whitespace-pre-line">
              {top.rules_text}
            </div>
          )}
        </Link>

        {/* 今日市场情绪 */}
        <div className="rounded-lg bg-gray-900 border border-gray-800 p-4">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-gray-500">
            <Activity size={14} /> 今日市场情绪
          </div>
          <div className="mt-3 flex items-baseline gap-3">
            <span className="text-5xl font-bold tabular-nums" style={{ color: fngInfo.color }}>
              {fngLast?.value ?? '—'}
            </span>
            <span className="text-sm text-gray-300">
              {fngLast?.classification ?? '—'}
            </span>
          </div>
          <div className="text-xs mt-2" style={{ color: fngInfo.color }}>
            {fngInfo.label}
          </div>
          <div className="text-[11px] text-gray-500 mt-3">
            数据源：alternative.me · 恐慌贪婪指数 0–100
          </div>
        </div>

        {/* 策略概览 */}
        <Link
          to="/rankings"
          className="rounded-lg bg-gray-900 border border-gray-800 p-4 hover:border-sky-700 transition"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-gray-500">
              <BarChart3 size={14} /> 策略概览
            </div>
            <ArrowRight size={14} className="text-gray-500" />
          </div>
          <div className="grid grid-cols-2 gap-3 mt-3">
            <div>
              <div className="text-xs text-gray-500">策略总数</div>
              <div className="text-2xl font-semibold tabular-nums text-gray-100">{summary.total}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">平均夏普</div>
              <div className="text-2xl font-semibold tabular-nums text-sky-300">
                {summary.avgSharpe == null ? '—' : summary.avgSharpe.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500">盈利策略</div>
              <div className="text-xl font-semibold tabular-nums text-emerald-400">{summary.profitable}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500">亏损策略</div>
              <div className="text-xl font-semibold tabular-nums text-rose-400">{summary.losing}</div>
            </div>
          </div>
        </Link>
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">
          {top ? top.display_name : '最佳策略'} · {bestEquity?.symbol?.replace('USDT', '') || 'BTC'} 关键指标
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          <MetricCard label="总收益率" tone="up"
            value={fmtPct(top?.primary?.total_return_pct ?? null, 1)}
            sub={`期末 ${fmtNum(top?.primary?.final_equity, 2)} USDT`} />
          <MetricCard label="年化收益率" tone="up"
            value={fmtPct(top?.primary?.annualized_return_pct ?? null, 1)} />
          <MetricCard label="夏普比率" tone="accent"
            value={fmtNum(top?.primary?.sharpe_ratio, 2)} />
          <MetricCard label="最大回撤" tone="down"
            value={fmtPct(top?.primary?.max_drawdown_pct ?? null, 1)}
            sub={`熔断 ${top?.primary?.circuit_breaker ? '触发' : '未触发'}`} />
          <MetricCard label="胜率"
            value={fmtPct(top?.primary?.win_rate_pct ?? null, 1)}
            sub={`${top?.primary?.total_trades ?? '—'} 笔 · 盈亏比 ${fmtNum(top?.primary?.profit_loss_ratio, 2)}`} />
        </div>
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 rounded-lg bg-gray-900 border border-gray-800 p-4">
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-sm uppercase tracking-wider text-gray-400">
              净值曲线（左轴）vs {bestEquity?.symbol?.replace('USDT', '') || 'BTC'} 价格（右轴）
            </h2>
            <span className="text-xs text-gray-500">日采样</span>
          </div>
          <EquityChart points={bestEquity?.points || []} height={340} />
        </div>
        <div className="rounded-lg bg-gray-900 border border-gray-800 p-4">
          <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">
            策略迷你曲线（综合排名前 5）
          </h2>
          <div className="space-y-2">
            {(index?.strategies || [])
              .slice()
              .sort((a, b) => b.composite_score - a.composite_score)
              .slice(0, 5)
              .map((s, i) => (
                <Link
                  key={s.id}
                  to={`/backtest?strategy=${s.id}`}
                  className="flex items-center justify-between gap-2 px-2 py-1.5 rounded hover:bg-gray-800/40"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-xs text-gray-500 tabular-nums w-4">{i + 1}</span>
                    <span className="text-xs text-gray-200 truncate">{s.display_name}</span>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <Sparkline data={s.sparkline} width={80} height={20} />
                    <span className={`text-xs tabular-nums w-16 text-right ${s.primary.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {fmtPct(s.primary.total_return_pct, 0)}
                    </span>
                  </div>
                </Link>
              ))}
          </div>
        </div>
      </section>
    </div>
  )
}
