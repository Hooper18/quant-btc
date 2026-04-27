import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Trophy, Medal, Award } from 'lucide-react'
import Sparkline from '../components/Sparkline'
import { fmtPct, fmtNum } from '../utils/dataLoader'
import { loadStrategiesIndex } from '../utils/strategyData'

const TABS = [
  { key: 'composite', label: '综合排行', getter: (s) => s.composite_score },
  { key: 'return', label: '收益排行', getter: (s) => s.primary.total_return_pct },
  { key: 'sharpe', label: '夏普排行', getter: (s) => s.primary.sharpe_ratio },
]

const COLUMNS = [
  { key: 'rank', label: '#', width: 40 },
  { key: 'display_name', label: '策略', sortable: false },
  { key: 'category', label: '类别', sortable: false },
  { key: 'composite_score', label: '综合分', getter: (s) => s.composite_score, fmt: (v) => v?.toFixed(2) },
  { key: 'total_return_pct', label: '总收益', getter: (s) => s.primary.total_return_pct, fmt: (v) => fmtPct(v, 1) },
  { key: 'annualized_return_pct', label: '年化', getter: (s) => s.primary.annualized_return_pct, fmt: (v) => fmtPct(v, 1) },
  { key: 'sharpe_ratio', label: '夏普', getter: (s) => s.primary.sharpe_ratio, fmt: (v) => fmtNum(v, 2) },
  { key: 'max_drawdown_pct', label: '最大回撤', getter: (s) => s.primary.max_drawdown_pct, fmt: (v) => fmtPct(v, 1) },
  { key: 'win_rate_pct', label: '胜率', getter: (s) => s.primary.win_rate_pct, fmt: (v) => fmtPct(v, 1) },
  { key: 'total_trades', label: '笔数', getter: (s) => s.primary.total_trades, fmt: (v) => fmtNum(v, 0) },
]

function rankIcon(rank) {
  if (rank === 1) return <Trophy size={20} className="text-amber-400" />
  if (rank === 2) return <Medal size={20} className="text-gray-300" />
  if (rank === 3) return <Award size={20} className="text-orange-500" />
  return null
}

function rankBgClass(rank) {
  if (rank === 1) return 'border-amber-500/40 bg-gradient-to-br from-amber-950/40 to-gray-900'
  if (rank === 2) return 'border-gray-400/30 bg-gradient-to-br from-gray-800/40 to-gray-900'
  if (rank === 3) return 'border-orange-500/30 bg-gradient-to-br from-orange-950/30 to-gray-900'
  return 'border-gray-800 bg-gray-900'
}

export default function Rankings() {
  const [index, setIndex] = useState(null)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('composite')
  const [sortKey, setSortKey] = useState('composite_score')
  const [sortDir, setSortDir] = useState('desc')
  const navigate = useNavigate()

  useEffect(() => {
    loadStrategiesIndex().then(setIndex).catch((e) => setError(e.message))
  }, [])

  const tabConfig = TABS.find((t) => t.key === tab)
  const ranked = useMemo(() => {
    if (!index?.strategies) return []
    return [...index.strategies].sort((a, b) => tabConfig.getter(b) - tabConfig.getter(a))
  }, [index, tab])

  const tableSorted = useMemo(() => {
    if (!index?.strategies) return []
    const col = COLUMNS.find((c) => c.key === sortKey)
    if (!col?.getter) return [...index.strategies]
    const arr = [...index.strategies]
    arr.sort((a, b) => {
      const av = col.getter(a)
      const bv = col.getter(b)
      if (av === bv) return 0
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return arr
  }, [index, sortKey, sortDir])

  const onColClick = (col) => {
    if (!col.getter) return
    if (sortKey === col.key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(col.key)
      setSortDir('desc')
    }
  }

  const onCardClick = (s) => navigate(`/backtest?strategy=${s.id}`)

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-xl font-semibold">策略排行榜</h1>
        <span className="text-xs text-gray-500">{index?.count ?? '—'} 个策略 · BTC 主回测</span>
      </div>

      {error && (
        <div className="rounded bg-rose-950/40 border border-rose-900 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="flex flex-wrap gap-1 rounded-md bg-gray-900 border border-gray-800 p-1 w-fit">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-1.5 rounded text-sm transition ${
              tab === t.key ? 'bg-sky-600 text-white' : 'text-gray-400 hover:text-gray-100'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {ranked.map((s, i) => {
          const rank = i + 1
          return (
            <button
              key={s.id}
              onClick={() => onCardClick(s)}
              className={`text-left rounded-lg border p-4 transition hover:border-sky-700 hover:shadow-lg hover:shadow-sky-900/20 ${rankBgClass(rank)}`}
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  <div className="text-2xl font-bold tabular-nums text-gray-200 w-8">{rank}</div>
                  {rankIcon(rank)}
                </div>
                <div className="text-right">
                  <div className="text-[11px] uppercase tracking-wider text-gray-500">综合分</div>
                  <div className="text-lg font-semibold text-sky-300 tabular-nums">
                    {s.composite_score?.toFixed(2)}
                  </div>
                </div>
              </div>
              <div className="text-base font-semibold text-gray-100 mb-1">{s.display_name}</div>
              <div className="text-[11px] text-gray-500 mb-3">{s.category}</div>

              <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                <div>
                  <div className="text-gray-500">总收益</div>
                  <div className={`font-semibold tabular-nums ${s.primary.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {fmtPct(s.primary.total_return_pct, 1)}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">夏普</div>
                  <div className="font-semibold tabular-nums text-sky-300">
                    {fmtNum(s.primary.sharpe_ratio, 2)}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">最大回撤</div>
                  <div className="font-semibold tabular-nums text-amber-300">
                    {fmtPct(s.primary.max_drawdown_pct, 1)}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">胜率</div>
                  <div className="font-semibold tabular-nums text-gray-200">
                    {fmtPct(s.primary.win_rate_pct, 1)}
                  </div>
                </div>
              </div>
              <div className="border-t border-gray-800 pt-2">
                <Sparkline data={s.sparkline} width={260} height={36} />
              </div>
            </button>
          )
        })}
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-gray-400">完整对比表</h2>
          <span className="text-xs text-gray-500">点表头排序</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr>
                {COLUMNS.map((c) => (
                  <th
                    key={c.key}
                    onClick={() => onColClick(c)}
                    className={`px-2 py-2 text-left font-normal whitespace-nowrap ${
                      c.getter ? 'cursor-pointer hover:text-gray-300' : ''
                    } ${sortKey === c.key ? 'text-sky-400' : ''}`}
                    style={c.width ? { width: c.width } : undefined}
                  >
                    {c.label}
                    {sortKey === c.key && (sortDir === 'asc' ? ' ↑' : ' ↓')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableSorted.map((s, i) => (
                <tr
                  key={s.id}
                  className="border-b border-gray-900/60 hover:bg-gray-800/40 cursor-pointer"
                  onClick={() => onCardClick(s)}
                >
                  <td className="px-2 py-1.5 tabular-nums text-gray-500">{i + 1}</td>
                  <td className="px-2 py-1.5 text-gray-200 whitespace-nowrap">{s.display_name}</td>
                  <td className="px-2 py-1.5 text-gray-500 whitespace-nowrap">{s.category}</td>
                  <td className="px-2 py-1.5 tabular-nums text-sky-300">{s.composite_score?.toFixed(2)}</td>
                  <td className={`px-2 py-1.5 tabular-nums ${s.primary.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {fmtPct(s.primary.total_return_pct, 1)}
                  </td>
                  <td className={`px-2 py-1.5 tabular-nums ${s.primary.annualized_return_pct >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                    {fmtPct(s.primary.annualized_return_pct, 1)}
                  </td>
                  <td className="px-2 py-1.5 tabular-nums text-sky-300">
                    {fmtNum(s.primary.sharpe_ratio, 2)}
                  </td>
                  <td className="px-2 py-1.5 tabular-nums text-amber-300">
                    {fmtPct(s.primary.max_drawdown_pct, 1)}
                  </td>
                  <td className="px-2 py-1.5 tabular-nums text-gray-300">
                    {fmtPct(s.primary.win_rate_pct, 1)}
                  </td>
                  <td className="px-2 py-1.5 tabular-nums text-gray-400">
                    {fmtNum(s.primary.total_trades, 0)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <p className="text-[11px] text-gray-500">
        综合评分 = 0.3×标准化夏普 + 0.3×标准化年化收益 + 0.2×(1−标准化最大回撤) + 0.2×标准化胜率，
        所有指标按 BTC 主回测、min-max 归一化到 [0,100]。
      </p>
    </div>
  )
}
