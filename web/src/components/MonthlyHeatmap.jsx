const MONTH_LABELS = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

function colorFor(v) {
  if (v === null || v === undefined) return '#0b1220'
  const cap = 30
  const x = Math.max(-cap, Math.min(cap, v)) / cap
  if (x >= 0) {
    const a = 0.15 + 0.55 * x
    return `rgba(16,185,129,${a.toFixed(3)})`
  }
  const a = 0.15 + 0.55 * Math.abs(x)
  return `rgba(239,68,68,${a.toFixed(3)})`
}

export default function MonthlyHeatmap({ matrix }) {
  if (!matrix || matrix.length === 0) return <div className="text-gray-500 text-sm">无数据</div>
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 text-xs">
        <thead>
          <tr>
            <th className="w-12 text-gray-500 font-normal text-left px-2">年</th>
            {MONTH_LABELS.map((m) => (
              <th key={m} className="w-14 text-gray-500 font-normal">{m}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row) => (
            <tr key={row.year}>
              <td className="text-gray-400 px-2 tabular-nums">{row.year}</td>
              {row.months.map((v, i) => (
                <td
                  key={i}
                  className="rounded text-center tabular-nums"
                  style={{
                    backgroundColor: colorFor(v),
                    color: v === null || v === undefined ? '#374151' : '#f9fafb',
                    width: 56, height: 28, fontSize: 11,
                  }}
                  title={v !== null && v !== undefined ? `${row.year}-${i + 1}: ${v.toFixed(2)}%` : '—'}
                >
                  {v === null || v === undefined ? '—' : v.toFixed(1)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
