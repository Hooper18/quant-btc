import { useEffect, useRef, useState } from 'react'

const LABELS = { BTCUSDT: 'BTC', ETHUSDT: 'ETH', SOLUSDT: 'SOL' }

// 单一 WS 复用：连接合并 stream，分发到各订阅者
const SOCKET_URL = (symbols) =>
  `wss://stream.binance.com:9443/stream?streams=${symbols
    .map((s) => `${s.toLowerCase()}@ticker`)
    .join('/')}`

let ws = null
let listeners = new Set()

function ensureWs(symbols) {
  if (ws && (ws.readyState === 0 || ws.readyState === 1)) return ws
  ws = new WebSocket(SOCKET_URL(symbols))
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data)
      const d = msg.data
      if (!d) return
      const update = {
        symbol: d.s,
        price: parseFloat(d.c),
        change: parseFloat(d.p),
        changePct: parseFloat(d.P),
        high: parseFloat(d.h),
        low: parseFloat(d.l),
        volume: parseFloat(d.q),
      }
      listeners.forEach((fn) => fn(update))
    } catch {
      /* ignore */
    }
  }
  ws.onclose = () => {
    ws = null
    setTimeout(() => ensureWs(symbols), 3000)
  }
  return ws
}

export function useTicker(symbols) {
  const [tickers, setTickers] = useState({})
  const symRef = useRef(symbols.join(','))
  useEffect(() => {
    symRef.current = symbols.join(',')
    ensureWs(symbols)
    const fn = (update) => {
      setTickers((prev) => ({ ...prev, [update.symbol]: update }))
    }
    listeners.add(fn)
    return () => { listeners.delete(fn) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symRef.current])
  return tickers
}

export default function LivePrice({ symbol, size = 'lg' }) {
  const tickers = useTicker([symbol])
  const t = tickers[symbol]

  const containerCls = size === 'lg'
    ? 'rounded-xl bg-gray-900 border border-gray-800 px-5 py-4'
    : 'rounded-lg bg-gray-900 border border-gray-800 px-4 py-3'
  const priceCls = size === 'lg' ? 'text-3xl' : 'text-2xl'

  if (!t) {
    return (
      <div className={containerCls}>
        <div className="text-xs text-gray-500">{LABELS[symbol] || symbol}</div>
        <div className={`${priceCls} text-gray-600 mt-1 tabular-nums`}>—</div>
        <div className="text-[11px] text-gray-600 mt-1">连接中…</div>
      </div>
    )
  }
  const up = t.changePct >= 0
  return (
    <div className={containerCls}>
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-gray-400">{LABELS[symbol] || symbol}</span>
        <span className={`text-xs tabular-nums ${up ? 'text-emerald-400' : 'text-rose-400'}`}>
          {up ? '+' : ''}{t.changePct.toFixed(2)}%
        </span>
      </div>
      <div className={`${priceCls} font-semibold tabular-nums mt-1 ${up ? 'text-emerald-300' : 'text-rose-300'}`}>
        ${t.price.toLocaleString('en-US', { maximumFractionDigits: 2 })}
      </div>
      <div className="text-[11px] text-gray-500 mt-1 tabular-nums flex justify-between">
        <span>24h H {t.high.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
        <span>L {t.low.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
      </div>
    </div>
  )
}
