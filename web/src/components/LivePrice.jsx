import { useEffect, useState } from 'react'

const LABELS = { BTCUSDT: 'BTC', ETHUSDT: 'ETH', SOLUSDT: 'SOL' }

// 固定订阅 BTC/ETH/SOL 三路 ticker — 任意组件 useTicker 时复用同一连接，
// 避免"先 mount 的单币种把 WS 锁死，后挂载的拿不到数据"的问题。
const ALL_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
const STREAM_URL = `wss://stream.binance.com:9443/stream?streams=${ALL_SYMBOLS
  .map((s) => `${s.toLowerCase()}@ticker`)
  .join('/')}`

let ws = null
let listeners = new Set()
let lastSnapshot = {}

function ensureWs() {
  if (ws && (ws.readyState === 0 || ws.readyState === 1)) return ws
  ws = new WebSocket(STREAM_URL)
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
      lastSnapshot[update.symbol] = update
      listeners.forEach((fn) => fn(update))
    } catch {
      /* ignore */
    }
  }
  ws.onclose = () => {
    ws = null
    setTimeout(() => ensureWs(), 3000)
  }
  ws.onerror = () => {
    try { ws && ws.close() } catch { /* ignore */ }
  }
  return ws
}

export function useTicker(_symbols) {
  const [tickers, setTickers] = useState(() => ({ ...lastSnapshot }))
  useEffect(() => {
    ensureWs()
    if (Object.keys(lastSnapshot).length) {
      setTickers({ ...lastSnapshot })
    }
    const fn = (update) => {
      setTickers((prev) => ({ ...prev, [update.symbol]: update }))
    }
    listeners.add(fn)
    return () => { listeners.delete(fn) }
  }, [])
  return tickers
}

export default function LivePrice({ symbol, size = 'lg' }) {
  const tickers = useTicker(ALL_SYMBOLS)
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
