import { ExternalLink, BookOpen, Cpu, BarChart3 } from 'lucide-react'

function Section({ title, icon: Icon, children, className = '' }) {
  return (
    <section className={`rounded-lg bg-gray-900 border border-gray-800 p-5 ${className}`}>
      <h2 className="text-base text-gray-100 mb-3 flex items-center gap-2">
        {Icon && <Icon size={16} className="text-sky-400" />} {title}
      </h2>
      <div className="text-sm leading-7 text-gray-300 space-y-2">{children}</div>
    </section>
  )
}

function Term({ name, children }) {
  return (
    <div>
      <span className="text-gray-100 font-medium">{name}</span>
      <span className="text-gray-500"> — </span>
      <span>{children}</span>
    </div>
  )
}

export default function About() {
  return (
    <div className="space-y-6 max-w-3xl">
      <h1 className="text-xl font-semibold">关于本项目</h1>

      <Section title="符号主义规则引擎 BTC 量化系统">
        <p>
          quant-btc 用<strong className="text-sky-300">可读的规则与参数组合</strong>表达策略，避免黑盒模型；
          指标 → 信号 → 仓位 → 风控 → 可视化全链路可解释、可回溯。当前 MVP 已打通：
          数据管道（Binance Vision 月度 ZIP + Parquet 月度分区）、指标层（pandas-ta 15 个指标）、
          规则引擎（YAML 加载 + 跨周期对齐 + 信号方向冲突仲裁）、回测引擎（杠杆永续逐 K 线推进 +
          止损止盈/强平/最大回撤熔断/资金费率结算）、可视化报告。
        </p>
      </Section>

      <Section title="技术栈" icon={Cpu}>
        <ul className="list-disc list-inside space-y-1 text-gray-300">
          <li><span className="text-gray-100">回测引擎</span>：Python 3.12 · uv · Polars · pandas-ta</li>
          <li><span className="text-gray-100">数据存储</span>：Parquet 月度分区，按 <code className="bg-gray-950 px-1 py-0.5 rounded text-gray-300">{'{symbol}/{tf}_{YYYY}_{MM}.parquet'}</code></li>
          <li><span className="text-gray-100">前端</span>：React 19 · Vite · Tailwind v4 · Recharts · Lucide</li>
          <li><span className="text-gray-100">实时行情</span>：Binance Spot WebSocket（<code className="bg-gray-950 px-1 py-0.5 rounded text-gray-300">stream.binance.com:9443</code>）</li>
          <li><span className="text-gray-100">部署</span>：Vercel · 静态前端 + 月度 cron 触发数据更新</li>
        </ul>
      </Section>

      <Section title="指标术语" icon={BookOpen}>
        <Term name="夏普比率（Sharpe Ratio）">
          风险调整后收益。计算 = 年化收益 / 年化波动率。粗略口径：&gt;1 可用、&gt;2 优秀、&gt;3 接近过拟合需警惕。本项目按
          √(365×24) 把小时级收益年化。
        </Term>
        <Term name="最大回撤（Max Drawdown）">
          权益曲线从历史峰值跌到谷底的最大跌幅，体现"最坏体验"。回测里 NAV 跌穿 30% 会触发回撤熔断、强制平仓停止交易。
        </Term>
        <Term name="胜率（Win Rate）">
          盈利交易笔数 / 总平仓笔数。趋势策略胜率常在 30%–50%，但盈亏比高；均值回归策略胜率高但盈亏比偏低，需结合看。
        </Term>
        <Term name="盈亏比（Profit/Loss Ratio）">
          平均盈利 / 平均亏损（绝对值）。胜率 × 盈亏比 &gt; 1 才是正期望策略。
        </Term>
        <Term name="年化收益率（Annualized Return）">
          回测期间几何收益按 365 天年化的结果。复利效应下，长期年化更能反映真实表现。
        </Term>
        <Term name="综合评分">
          排行榜专用：0.3×标准化夏普 + 0.3×标准化年化 + 0.2×(1−标准化最大回撤) + 0.2×标准化胜率。所有指标按 BTC 主回测、min-max 归一到 [0,100]，便于跨策略横向比较。
        </Term>
      </Section>

      <Section title="策略规则引擎" icon={BarChart3}>
        <p>
          策略以 YAML 描述：每条策略由若干<strong>条件</strong>组合（AND/OR），命中后执行<strong>动作</strong>（开多/开空 + 仓位 + 止损止盈）。
        </p>
        <div className="space-y-1.5 text-gray-300">
          <Term name="支持的指标">
            趋势：SMA / EMA / MACD（line/signal/hist）/ ADX / Bollinger / Keltner； 动量：RSI / Stoch / CCI / Williams%R；
            量能：OBV / VWAP / MFI / CMF / taker_buy_ratio； 衍生：rolling_max/min、oi_change_N、fear_greed_ma_N。
          </Term>
          <Term name="支持的条件类型">
            阈值（{'>'}/{'<'}/=/{'>'}=/{'<='}）、交叉（cross above/below 某参考线）、状态记忆（from_above/to_below 区间穿越）、嵌套 AND/OR 子条件。
          </Term>
          <Term name="跨周期">
            每个条件独立 timeframe，例如 1h RSI + 4h MACD 复合。回测引擎按主周期推进，向下取最近一根高周期 K 线对齐。
          </Term>
          <Term name="冲突仲裁">
            同一时刻命中多个反向信号时，引擎按"先平后开"原则处理，避免双向持仓。
          </Term>
        </div>
      </Section>

      <Section title="使用说明">
        <Term name="排行榜">
          点 "排行榜" 进入。3 个 tab：综合 / 收益 / 夏普。前 3 名有金/银/铜标识，每张卡含 sparkline 迷你净值图，点击跳到对应策略的回测详情。底部完整对比表支持点表头排序。
        </Term>
        <Term name="回测详情">
          顶部下拉切换策略。BTC/ETH/SOL 切换钮显示哪些币种数据可用（无 OI/FNG 的币种会被自动隐藏）。策略信息卡列出做多/做空条件、止损止盈、仓位。下方依次是 6 个指标卡 → 净值曲线 → 月度热力图 → 100 笔交易表。
        </Term>
        <Term name="风险分析">
          基于 V2 最优策略：1000 次蒙特卡洛 bootstrap 给出收益/回撤分布，杠杆扫描给破产概率（建议 7×），参数敏感度热力图体现稳健区间。
        </Term>
        <Term name="实时行情">
          BTC/ETH/SOL 24h 价格 + 涨跌 + 成交额，恐慌指数最近 30 天。所有数据走 Binance 公共 WebSocket，无需 API key。
        </Term>
      </Section>

      <Section title="关键产物">
        <ul className="list-disc list-inside space-y-1">
          <li>BTC v2 最优策略：总收益 +2012% / 年化 +63% / 夏普 2.26 / 最大回撤 21%（6 年回测）</li>
          <li>Walk-Forward 20 窗口验证：年化 +40% / 夏普 1.33（样本外）</li>
          <li>蒙特卡洛 1000 次 bootstrap：建议 7× 杠杆，破产概率 &lt; 5%</li>
          <li>三币种组合（BTC 50% / ETH 25% / SOL 25%）：年化 +43% / 夏普 1.99</li>
          <li>10 套独立策略覆盖趋势跟踪 / 均值回归 / 情绪反向 / OI 背离 / 多周期复合等思路</li>
        </ul>
      </Section>

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
