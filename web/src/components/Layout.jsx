import { NavLink, Outlet } from 'react-router-dom'
import { LayoutDashboard, LineChart, ShieldAlert, Activity, Info } from 'lucide-react'

const NAV = [
  { to: '/',        label: '概览',      icon: LayoutDashboard },
  { to: '/backtest',label: '回测详情',  icon: LineChart },
  { to: '/risk',    label: '风险分析',  icon: ShieldAlert },
  { to: '/market',  label: '实时行情',  icon: Activity },
  { to: '/about',   label: '关于',      icon: Info },
]

export default function Layout() {
  return (
    <div className="min-h-screen flex bg-gray-950 text-gray-100">
      <aside className="w-56 shrink-0 border-r border-gray-800 bg-gray-950 flex flex-col">
        <div className="px-5 py-5 border-b border-gray-800">
          <div className="text-lg font-semibold tracking-tight">quant-btc</div>
          <div className="text-xs text-gray-500 mt-1">符号主义量化引擎</div>
        </div>
        <nav className="flex-1 px-2 py-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
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
        <div className="px-5 py-3 text-[11px] text-gray-600 border-t border-gray-800">
          v2 最优策略 · 7× 杠杆建议
        </div>
      </aside>
      <main className="flex-1 min-w-0 overflow-x-hidden">
        <header className="sticky top-0 z-10 bg-gray-950/85 backdrop-blur border-b border-gray-800 px-6 h-12 flex items-center justify-between">
          <div className="text-sm text-gray-400">BTC/ETH/SOL 永续合约 · 符号主义规则引擎</div>
          <div className="text-xs text-gray-500">本系统仅供学习研究，不构成投资建议</div>
        </header>
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
