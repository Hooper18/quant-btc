// 加载 public/data/ 下的 JSON。Vite 会按 base URL 处理。
const cache = new Map()

export async function loadJson(name) {
  if (cache.has(name)) return cache.get(name)
  const url = `${import.meta.env.BASE_URL}data/${name}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`加载失败: ${name} (${res.status})`)
  const data = await res.json()
  cache.set(name, data)
  return data
}

export function fmtPct(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${Number(v).toFixed(digits)}%`
}

export function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return Number(v).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function fmtCompact(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const n = Number(v)
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + 'K'
  return n.toFixed(2)
}

export function classFor(value, positive = '#10b981', negative = '#ef4444', zero = '#9ca3af') {
  if (value === null || value === undefined || Number.isNaN(value)) return zero
  if (value > 0) return positive
  if (value < 0) return negative
  return zero
}
