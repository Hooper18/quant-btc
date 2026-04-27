import { useEffect, useState } from 'react'
import LivePrice, { useTicker } from '../components/LivePrice'
import { loadJson } from '../utils/dataLoader'

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

function fngColor(value) {
  if (value <= 24) return '#dc2626'
  if (value <= 49) return '#f59e0b'
  if (value <= 54) return '#facc15'
  if (value <= 74) return '#84cc16'
  return '#22c55e'
}

export default function MarketData() {
  const tickers = useTicker(SYMBOLS)
  const [fng, setFng] = useState(null)
  useEffect(() => {
    loadJson('fear_greed_latest.json').then(setFng).catch(() => {})
  }, [])

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">实时行情</h1>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {SYMBOLS.map((s) => (
          <LivePrice key={s} symbol={s} />
        ))}
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <h2 className="text-sm uppercase tracking-wider text-gray-400 mb-3">24h 详细</h2>
        <table className="w-full text-xs">
          <thead className="text-gray-500 border-b border-gray-800">
            <tr>
              {['币种', '最新价', '24h 涨跌', '24h 涨跌%', '24h 高', '24h 低', '24h 成交额'].map((h) => (
                <th key={h} className="px-2 py-1.5 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SYMBOLS.map((s) => {
              const t = tickers[s]
              if (!t) {
                return (
                  <tr key={s} className="border-b border-gray-900/60">
                    <td className="px-2 py-1.5 text-gray-200">{s.replace('USDT','')}</td>
                    <td colSpan={6} className="px-2 py-1.5 text-gray-500">连接中…</td>
                  </tr>
                )
              }
              const up = t.changePct >= 0
              return (
                <tr key={s} className="border-b border-gray-900/60">
                  <td className="px-2 py-1.5 text-gray-200">{s.replace('USDT','')}</td>
                  <td className={`px-2 py-1.5 tabular-nums ${up ? 'text-emerald-400' : 'text-rose-400'}`}>
                    ${t.price.toLocaleString('en-US', { maximumFractionDigits: 2 })}
                  </td>
                  <td className={`px-2 py-1.5 tabular-nums ${up ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {up ? '+' : ''}{t.change.toFixed(2)}
                  </td>
                  <td className={`px-2 py-1.5 tabular-nums ${up ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {up ? '+' : ''}{t.changePct.toFixed(2)}%
                  </td>
                  <td className="px-2 py-1.5 tabular-nums text-gray-300">${t.high.toLocaleString('en-US', { maximumFractionDigits: 2 })}</td>
                  <td className="px-2 py-1.5 tabular-nums text-gray-300">${t.low.toLocaleString('en-US', { maximumFractionDigits: 2 })}</td>
                  <td className="px-2 py-1.5 tabular-nums text-gray-400">${(t.volume / 1e6).toFixed(2)}M</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="text-[11px] text-gray-500 mt-3">
          数据源：Binance Spot WebSocket（<code className="bg-gray-950 px-1 py-0.5 rounded">wss://stream.binance.com:9443</code>）
        </div>
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-gray-400">恐慌指数（最近 30 天）</h2>
          <span className="text-xs text-gray-500">数据源：alternative.me</span>
        </div>
        <div className="flex items-end gap-1 h-32">
          {(fng?.rows || []).map((r) => (
            <div
              key={r.t}
              className="flex-1 rounded-t"
              style={{ height: `${Math.max(4, r.value)}%`, backgroundColor: fngColor(r.value) }}
              title={`${r.t}: ${r.value} (${r.classification})`}
            />
          ))}
        </div>
        <div className="flex justify-between text-[10px] text-gray-500 mt-1">
          <span>{fng?.rows?.[0]?.t}</span>
          <span>{fng?.rows?.[fng?.rows?.length - 1]?.t}</span>
        </div>
        {fng?.rows?.length > 0 && (
          <div className="mt-3 text-sm">
            <span className="text-gray-400">最新：</span>
            <span className="tabular-nums" style={{ color: fngColor(fng.rows[fng.rows.length - 1].value) }}>
              {fng.rows[fng.rows.length - 1].value} · {fng.rows[fng.rows.length - 1].classification}
            </span>
          </div>
        )}
      </section>
    </div>
  )
}
