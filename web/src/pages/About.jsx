import { ExternalLink } from 'lucide-react'

export default function About() {
  return (
    <div className="space-y-6 max-w-3xl">
      <h1 className="text-xl font-semibold">关于本项目</h1>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-5 space-y-3 text-sm leading-7 text-gray-300">
        <h2 className="text-lg text-gray-100">符号主义规则引擎 BTC 量化系统</h2>
        <p>
          quant-btc 用<strong className="text-sky-300">可读的规则与参数组合</strong>表达策略，避免黑盒模型；
          指标 → 信号 → 仓位 → 风控 → 可视化全链路可解释、可回溯。当前 MVP 已打通：
          数据管道（Binance Vision 月度 ZIP + Parquet 月度分区）、指标层（pandas-ta 15 个指标）、
          规则引擎（YAML 加载 + 跨周期对齐 + 信号方向冲突仲裁）、回测引擎（杠杆永续逐 K 线推进 +
          止损止盈/强平/最大回撤熔断/资金费率结算）、可视化报告。
        </p>
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-5 space-y-2 text-sm text-gray-300">
        <h2 className="text-base text-gray-100 mb-2">技术栈</h2>
        <ul className="list-disc list-inside space-y-1">
          <li><span className="text-gray-100">回测引擎</span>：Python 3.12 · uv · Polars · pandas-ta</li>
          <li><span className="text-gray-100">数据存储</span>：Parquet 月度分区，按 <code className="bg-gray-950 px-1 py-0.5 rounded text-gray-300">{'{symbol}/{tf}_{YYYY}_{MM}.parquet'}</code></li>
          <li><span className="text-gray-100">前端</span>：React 19 · Vite · Tailwind v4 · Recharts · Lucide</li>
          <li><span className="text-gray-100">实时行情</span>：Binance Spot WebSocket（<code className="bg-gray-950 px-1 py-0.5 rounded text-gray-300">stream.binance.com:9443</code>）</li>
          <li><span className="text-gray-100">部署</span>：Vercel · 静态前端 + 月度 cron 触发数据更新</li>
        </ul>
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-5 space-y-2 text-sm text-gray-300">
        <h2 className="text-base text-gray-100 mb-2">关键产物</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>BTC v2 最优策略：总收益 +2012% / 年化 +63% / 夏普 2.26 / 最大回撤 21%（6 年回测）</li>
          <li>Walk-Forward 20 窗口验证：年化 +40% / 夏普 1.33（样本外）</li>
          <li>蒙特卡洛 1000 次 bootstrap：建议 7× 杠杆，破产概率 &lt; 5%</li>
          <li>三币种组合（BTC 50% / ETH 25% / SOL 25%）：年化 +43% / 夏普 1.99</li>
        </ul>
      </section>

      <section className="rounded-lg bg-amber-950/30 border border-amber-900/50 p-5 text-sm text-amber-200 space-y-2">
        <h2 className="text-base text-amber-100">免责声明</h2>
        <p>
          本系统仅供学习研究使用，<strong>不构成任何投资建议</strong>。加密货币合约市场风险极高，
          可能在短时间内损失全部本金。回测表现源于历史数据，<strong>不能保证未来收益</strong>，
          且不计入交易所规则变更、深度不足、API 中断等真实场景因素。任何依据本系统进行的实盘操作，
          后果自负。
        </p>
      </section>

      <section className="rounded-lg bg-gray-900 border border-gray-800 p-5 text-sm">
        <a
          href="https://github.com/"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 text-sky-400 hover:text-sky-300"
        >
          <ExternalLink size={16} /> GitHub 仓库
        </a>
      </section>
    </div>
  )
}
