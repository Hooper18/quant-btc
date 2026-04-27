import { useMemo, useState } from 'react'

const SIDE_LABEL = {
  long_open: '开多', long_close: '平多',
  short_open: '开空', short_close: '平空',
  liquidate: '强平',
}

function sideColor(side) {
  if (side?.startsWith('long')) return 'text-emerald-400'
  if (side?.startsWith('short')) return 'text-rose-400'
  if (side === 'liquidate') return 'text-amber-400'
  return 'text-gray-300'
}

export default function TradeTable({ rows, pageSize = 20, compact = false }) {
  const [page, setPage] = useState(0)
  const [sortKey, setSortKey] = useState('t')
  const [sortDir, setSortDir] = useState('desc')

  const sorted = useMemo(() => {
    const arr = [...(rows || [])]
    arr.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey]
      if (av === bv) return 0
      const cmp = av < bv ? -1 : 1
      return sortDir === 'asc' ? cmp : -cmp
    })
    return arr
  }, [rows, sortKey, sortDir])

  const total = sorted.length
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const slice = sorted.slice(page * pageSize, page * pageSize + pageSize)

  function flip(key) {
    if (key === sortKey) setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  if (!rows || rows.length === 0) {
    return <div className="text-gray-500 text-sm">无交易记录</div>
  }

  const cell = compact ? 'px-2 py-1.5' : 'px-3 py-2'

  return (
    <div className="text-xs">
      <table className="w-full">
        <thead className="text-gray-500 border-b border-gray-800">
          <tr>
            {[
              ['t', '时间'],
              ['side', '方向'],
              ['price', '价格'],
              ['size', '数量'],
              ['pnl', '盈亏'],
            ].map(([k, label]) => (
              <th
                key={k}
                onClick={() => flip(k)}
                className={`${cell} text-left font-normal cursor-pointer select-none hover:text-gray-200`}
              >
                {label}{sortKey === k ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {slice.map((r, i) => (
            <tr key={i} className="border-b border-gray-900/60 hover:bg-gray-900/40">
              <td className={`${cell} text-gray-400 tabular-nums whitespace-nowrap`}>
                {r.t.slice(0, 16).replace('T', ' ')}
              </td>
              <td className={`${cell} ${sideColor(r.side)}`}>
                {SIDE_LABEL[r.side] || r.side}
              </td>
              <td className={`${cell} text-gray-200 tabular-nums`}>
                {r.price.toLocaleString('en-US', { maximumFractionDigits: 2 })}
              </td>
              <td className={`${cell} text-gray-400 tabular-nums`}>
                {r.size.toFixed(4)}
              </td>
              <td
                className={`${cell} tabular-nums ${
                  r.pnl > 0 ? 'text-emerald-400' : r.pnl < 0 ? 'text-rose-400' : 'text-gray-400'
                }`}
              >
                {r.pnl > 0 ? '+' : ''}
                {r.pnl.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-3 text-gray-500">
          <span>共 {total} 笔 · 第 {page + 1} / {totalPages} 页</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              className="px-2 py-1 rounded border border-gray-800 hover:bg-gray-900 disabled:opacity-40"
            >
              上一页
            </button>
            <button
              onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 rounded border border-gray-800 hover:bg-gray-900 disabled:opacity-40"
            >
              下一页
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
