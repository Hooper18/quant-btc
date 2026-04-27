import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Backtest from './pages/Backtest'
import RiskAnalysis from './pages/RiskAnalysis'
import MarketData from './pages/MarketData'
import About from './pages/About'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/risk" element={<RiskAnalysis />} />
        <Route path="/market" element={<MarketData />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
