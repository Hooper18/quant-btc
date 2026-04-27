function colorFor(v, range) {
  if (v === null || v === undefined) return '#0b1220'
  const x = Math.max(-range, Math.min(range, v)) / range
  if (x >= 0) {
    const a = 0.18 + 0.6 * x
    return `rgba(16,185,129,${a.toFixed(3)})`
  }
  const a = 0.18 + 0.6 * Math.abs(x)
  return `rgba(239,68,68,${a.toFixed(3)})`
}

export default function SensitivityMap({ grid }) {
  if (!grid) return null
  const { x_label, x_values, y_label, y_values, matrix, best_x, best_y, name } = grid
  const flat = matrix.flat().filter((v) => v !== null && v !== undefined)
  const max = flat.length > 0 ? Math.max(...flat.map(Math.abs), 1) : 1

  return (
    <div className="rounded-lg bg-gray-900 border border-gray-800 p-4">
      <div className="text-sm text-gray-200 mb-2">
        <span className="text-gray-400">{name}：</span>{y_label} × {x_label}（夏普）
      </div>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1 text-xs">
          <thead>
            <tr>
              <th className="px-2 text-gray-500 font-normal text-right">{y_label}↓ / {x_label}→</th>
              {x_values.map((v) => (
                <th key={v} className="text-gray-400 font-normal w-14 tabular-nums">{v}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {y_values.map((yv, i) => (
              <tr key={yv}>
                <td className="px-2 text-gray-400 text-right tabular-nums">{yv}</td>
                {x_values.map((xv, j) => {
                  const v = matrix[i][j]
                  const isBest = yv === best_y && xv === best_x
                  return (
                    <td
                      key={j}
                      className="rounded text-center tabular-nums"
                      style={{
                        backgroundColor: colorFor(v, max),
                        color: '#f9fafb',
                        width: 56, height: 28, fontSize: 11,
                        outline: isBest ? '2px solid #fde047' : 'none',
                        outlineOffset: -2,
                      }}
                      title={`${y_label}=${yv} × ${x_label}=${xv}: ${v ?? '—'}`}
                    >
                      {v === null || v === undefined ? '—' : v.toFixed(2)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-[11px] text-gray-500 mt-2">
        ★ 黄框为当前最优参数（{y_label}={best_y}, {x_label}={best_x}）
      </div>
    </div>
  )
}
