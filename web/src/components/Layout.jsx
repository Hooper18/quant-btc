import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import {
  LayoutDashboard, Trophy, LineChart, ShieldAlert, Activity, Info, Menu, X,
} from 'lucide-react'
import { loadStrategiesIndex } from '../utils/strategyData'

const NAV = [
  { to: '/',         label: '概览',     icon: LayoutDashboard },
  { to: '/rankings', label: '排行榜',   icon: Trophy },
  { to: '/backtest', label: '回测详情', icon: LineChart },
  { to: '/risk',     label: '风险分析', icon: ShieldAlert },
  { to: '/market',   label: '实时行情', icon: Activity },
  { to: '/about',    label: '关于',     icon: Info },
]

function formatYearMonth(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}`
}

export default function Layout() {
  const [open, setOpen] = useState(false)
  const [versionInfo, setVersionInfo] = useState({ count: null, ym: null })

  useEffect(() => {
    loadStrategiesIndex()
      .then((idx) =>
        setVersionInfo({ count: idx.count, ym: formatYearMonth(idx.generated_at) }),
      )
      .catch(() => {})
  }, [])

  const close = () => setOpen(false)

  return (
    <div className="min-h-screen flex bg-gray-950 text-gray-100">
      {/* 移动端遮罩 */}
      {open && (
        <div
          className="fixed inset-0 bg-black/60 z-30 md:hidden"
          onClick={close}
        />
      )}

      {/* 侧边栏 */}
      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-40
          w-56 shrink-0 border-r border-gray-800 bg-gray-950 flex flex-col
          transform transition-transform duration-200
          ${open ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        <div className="px-5 py-5 border-b border-gray-800 flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold tracking-tight">quant-btc</div>
            <div className="text-xs text-gray-500 mt-1">符号主义量化引擎</div>
          </div>
          <button
            className="md:hidden text-gray-400 hover:text-gray-100"
            onClick={close}
            aria-label="关闭菜单"
          >
            <X size={18} />
          </button>
        </div>
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={close}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition ${
                  isActive
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-400 hover:text-gray-100 hover:bg-gray-900'
                }`
              }
            >
              <Icon size={16} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="px-5 py-3 text-[11px] text-gray-600 border-t border-gray-800 leading-relaxed">
          {versionInfo.count != null
            ? `${versionInfo.count} 个策略 · 数据截至 ${versionInfo.ym}`
            : '加载中…'}
        </div>
      </aside>

      <main className="flex-1 min-w-0 overflow-x-hidden">
        <header className="sticky top-0 z-10 bg-gray-950/85 backdrop-blur border-b border-gray-800 px-4 md:px-6 h-12 flex items-center justify-between gap-3">
          <button
            className="md:hidden text-gray-400 hover:text-gray-100"
            onClick={() => setOpen(true)}
            aria-label="打开菜单"
          >
            <Menu size={20} />
          </button>
          <div className="text-sm text-gray-400 truncate">
            BTC/ETH/SOL 永续合约 · 符号主义规则引擎
          </div>
          <div className="text-xs text-gray-500 hidden sm:block">
            本系统仅供学习研究，不构成投资建议
          </div>
        </header>
        <div className="p-4 md:p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
