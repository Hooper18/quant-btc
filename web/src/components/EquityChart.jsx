import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts'

export default function EquityChart({ points, height = 320, showPrice = true, navLabel = 'NAV' }) {
  if (!points || points.length === 0) {
    return <div className="text-gray-500 text-sm">无数据</div>
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 4 }}>
        <defs>
          <linearGradient id="navFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.45} />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="2 4" stroke="#1f2937" />
        <XAxis
          dataKey="t"
          stroke="#6b7280"
          tick={{ fill: '#9ca3af', fontSize: 11 }}
          tickFormatter={(v) => v.slice(2, 7)}
          minTickGap={50}
        />
        <YAxis
          yAxisId="nav"
          stroke="#38bdf8"
          tick={{ fill: '#9ca3af', fontSize: 11 }}
          tickFormatter={(v) => v.toLocaleString()}
          width={70}
        />
        {showPrice && (
          <YAxis
            yAxisId="price"
            orientation="right"
            stroke="#a78bfa"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            tickFormatter={(v) => v.toLocaleString()}
            width={70}
          />
        )}
        <Tooltip
          contentStyle={{
            backgroundColor: '#0b1220', border: '1px solid #1f2937',
            borderRadius: 6, color: '#e5e7eb',
          }}
          labelStyle={{ color: '#9ca3af' }}
          formatter={(value, name) => {
            if (typeof value !== 'number') return ['—', name]
            return [value.toLocaleString('en-US', { maximumFractionDigits: 2 }), name]
          }}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
        <Area
          yAxisId="nav" type="monotone" dataKey="nav" name={navLabel}
          stroke="#38bdf8" fill="url(#navFill)" strokeWidth={1.6} dot={false}
          isAnimationActive={false}
        />
        {showPrice && (
          <Line
            yAxisId="price" type="monotone" dataKey="price" name="价格"
            stroke="#a78bfa" strokeWidth={1.2} dot={false}
            isAnimationActive={false}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
