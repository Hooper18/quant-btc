import { useEffect, useState } from 'react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts'
import MetricCard from '../components/MetricCard'
import SensitivityMap from '../components/SensitivityMap'
import { loadJson, fmtPct, fmtNum } from '../utils/dataLoader'

function ruinColor(p) {
  if (p < 0.05) return '#10b981'
  if (p < 0.15) return '#fbbf24'
  if (p < 0.30) return '#f97316'
  return '#ef4444'
}

export default function RiskAnalysis() {
  const [lev, setLev] = useState(null)
  const [mc, setMc] = useState(null)
  const [sens, setSens] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      loadJson('leverage_scan.json'),
      loadJson('monte_carlo.json'),
      loadJson('sensitivity.json'),
    ])
      .then(([l, m, s]) => { setLev(l); setMc(m); setSens(s) })
      .catch((e) => setError(e.message))
  }, [])

  // Monte Carlo 收益直方图数据
  const mcReturnBuckets = mc?.return_histogram
    ? mc.return_histogram.counts.map((c, i) => ({
        x: ((mc.return_histogram.edges[i] + mc.return_histogram.edges[i + 1]) / 2).toFixed(0),
        count: c,
      }))
    : []

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">风险分析</h1>

      {error && (
        <div className="rounded bg-rose-950/40 border border-rose-900 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          label="建议杠杆"
          value={lev?.recommended_leverage != null ? `${lev.recommended_leverage}×` : '—'}
          tone="accent"
          sub="破产概率 < 5%"
        />
        <MetricCard
          label="MC 收益中位数"
          value={fmtPct(mc?.return_pct_percentiles?.['50'], 0)}
          tone="up"
          sub="1000 次 bootstrap"
        />
        <MetricCard
          label="MC 95% 置信收益"
          value={`${fmtPct(mc?.return_pct_percentiles?.['5'], 0)} ~ ${fmtPct(mc?.return_pct_percentiles?.['95'], 0)}`}
        />
        <MetricCard
          label="10× 破产概率"
          value={fmtPct((mc?.ruin_probability ?? 0) * 100, 1)}
          tone="down"
          sub={`阈值 NAV < ${fmtNum(mc?.ruin_threshold, 0)} USDT`}
        />
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-gray-400">
            杠杆扫描 · 破产概率 vs 年化收益
          </h2>
          <span className="text-xs text-gray-500">绿 &lt; 5% · 黄 &lt; 15% · 橙 &lt; 30% · 红 ≥ 30%</span>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={lev?.rows || []}>
            <CartesianGrid strokeDasharray="2 4" stroke="#1f2937" />
            <XAxis dataKey="leverage" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 11 }}
              label={{ value: '杠杆倍数', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
            <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 11 }}
              tickFormatter={(v) => (v * 100).toFixed(0) + '%'}
              label={{ value: '破产概率', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0b1220', border: '1px solid #1f2937', borderRadius: 6, color: '#e5e7eb' }}
              formatter={(v, n, p) => {
                if (n === 'ruin_probability') return [(v * 100).toFixed(2) + '%', '破产概率']
                return [v, n]
              }}
              labelFormatter={(l) => `杠杆 ${l}×`}
            />
            <ReferenceLine y={0.05} stroke="#10b981" strokeDasharray="3 3" />
            <Bar dataKey="ruin_probability">
              {(lev?.rows || []).map((r, i) => (
                <Cell key={i} fill={ruinColor(r.ruin_probability)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <table className="w-full text-xs mt-4">
          <thead className="text-gray-500 border-b border-gray-800">
            <tr>
              {['杠杆', '年化', '夏普', '最大回撤', '胜率', '破产概率'].map((h) => (
                <th key={h} className="px-2 py-1.5 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(lev?.rows || []).map((r) => (
              <tr key={r.leverage} className="border-b border-gray-900/60">
                <td className="px-2 py-1.5 tabular-nums text-gray-200">{r.leverage}×</td>
                <td className="px-2 py-1.5 tabular-nums text-emerald-400">{r.annualized_return_pct.toFixed(2)}%</td>
                <td className="px-2 py-1.5 tabular-nums text-sky-300">{r.sharpe_ratio.toFixed(2)}</td>
                <td className="px-2 py-1.5 tabular-nums text-amber-300">{r.max_drawdown_pct.toFixed(2)}%</td>
                <td className="px-2 py-1.5 tabular-nums text-gray-300">{r.win_rate_pct.toFixed(2)}%</td>
                <td className="px-2 py-1.5 tabular-nums" style={{ color: ruinColor(r.ruin_probability) }}>
                  {(r.ruin_probability * 100).toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">
          蒙特卡洛收益分布（1000 次 bootstrap）
        </h2>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={mcReturnBuckets}>
            <CartesianGrid strokeDasharray="2 4" stroke="#1f2937" />
            <XAxis dataKey="x" stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 11 }}
              label={{ value: '最终收益 %', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
            <YAxis stroke="#6b7280" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: '#0b1220', border: '1px solid #1f2937', borderRadius: 6, color: '#e5e7eb' }} />
            <Bar dataKey="count" fill="#38bdf8" />
          </BarChart>
        </ResponsiveContainer>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-3 text-xs">
          {mc &&
            ['5', '25', '50', '75', '95'].map((q) => (
              <div key={q} className="rounded bg-gray-950 border border-gray-800 px-3 py-2">
                <div className="text-gray-500">P{q} 收益</div>
                <div className="text-emerald-300 tabular-nums">
                  {mc.return_pct_percentiles[q]?.toFixed(1)}%
                </div>
                <div className="text-gray-500 mt-0.5">P{q} 回撤</div>
                <div className="text-rose-300 tabular-nums">
                  {mc.max_dd_pct_percentiles[q]?.toFixed(1)}%
                </div>
              </div>
            ))}
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-sm uppercase tracking-wider text-gray-400">参数敏感度（夏普热力图）</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {(sens?.grids || []).map((g) => (
            <SensitivityMap key={g.name} grid={g} />
          ))}
        </div>
      </section>
    </div>
  )
}
