export default function MetricCard({ label, value, suffix = '', tone = 'default', sub }) {
  const toneClass = {
    up: 'text-emerald-400',
    down: 'text-rose-400',
    accent: 'text-sky-400',
    default: 'text-gray-100',
  }[tone] || 'text-gray-100'
  return (
    <div className="rounded-lg bg-gray-900 border border-gray-800 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${toneClass}`}>
        {value}
        {suffix && <span className="text-base font-normal text-gray-400 ml-0.5">{suffix}</span>}
      </div>
      {sub && <div className="text-[11px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  )
}
