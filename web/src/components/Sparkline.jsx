export default function Sparkline({ data, width = 120, height = 32, stroke = '#38bdf8' }) {
  if (!data || data.length < 2) {
    return <div style={{ width, height }} className="text-gray-700 text-[10px]">—</div>
  }
  const lo = Math.min(...data)
  const hi = Math.max(...data)
  const span = hi - lo || 1
  const stepX = width / (data.length - 1)
  const points = data
    .map((v, i) => `${(i * stepX).toFixed(2)},${(height - ((v - lo) / span) * height).toFixed(2)}`)
    .join(' ')
  const last = data[data.length - 1]
  const first = data[0]
  const up = last >= first
  const color = up ? '#10b981' : '#ef4444'

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        fill="none"
        stroke={stroke || color}
        strokeWidth={1.4}
        strokeLinejoin="round"
        strokeLinecap="round"
        points={points}
      />
    </svg>
  )
}
