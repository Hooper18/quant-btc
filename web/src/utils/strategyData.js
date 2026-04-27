import { loadJson } from './dataLoader'

let indexCachePromise = null

export function loadStrategiesIndex() {
  if (!indexCachePromise) {
    indexCachePromise = loadJson('strategies_index.json')
  }
  return indexCachePromise
}

export function loadStrategyDetail(id) {
  return loadJson(`strategies/${id}.json`)
}

export function rankBy(strategies, key, opts = {}) {
  const { ascending = false, getter } = opts
  const sorted = [...strategies].sort((a, b) => {
    const av = getter ? getter(a) : a.primary[key]
    const bv = getter ? getter(b) : b.primary[key]
    if (av === bv) return 0
    return ascending ? av - bv : bv - av
  })
  return sorted
}

export function bestStrategy(index) {
  if (!index?.strategies?.length) return null
  return [...index.strategies].sort(
    (a, b) => b.composite_score - a.composite_score
  )[0]
}

export function strategySummary(index) {
  if (!index?.strategies?.length) {
    return { total: 0, profitable: 0, losing: 0, avgSharpe: null }
  }
  const items = index.strategies
  const profitable = items.filter((s) => s.primary.total_return_pct > 0).length
  const losing = items.filter((s) => s.primary.total_return_pct < 0).length
  const sharpes = items.map((s) => s.primary.sharpe_ratio)
  const avgSharpe = sharpes.reduce((a, b) => a + b, 0) / sharpes.length
  return { total: items.length, profitable, losing, avgSharpe }
}
