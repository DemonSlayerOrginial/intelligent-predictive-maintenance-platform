import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws')

function pct(value) {
  return `${Math.round((value || 0) * 100)}%`
}

function hours(value) {
  if (value === undefined || value === null) return '—'
  if (value < 24) return `${Math.round(value)}h`
  return `${(value / 24).toFixed(1)}d`
}

function RiskBadge({ level }) {
  return <span className={`badge ${level}`}>{level}</span>
}

function MetricCard({ label, value, detail }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

function App() {
  const [machines, setMachines] = useState([])
  const [summary, setSummary] = useState({})
  const [connected, setConnected] = useState(false)
  const [selected, setSelected] = useState(null)

  async function refresh() {
    try {
      const [machinesRes, summaryRes] = await Promise.all([
        fetch(`${API}/api/machines`),
        fetch(`${API}/api/summary`),
      ])
      if (machinesRes.ok) setMachines(await machinesRes.json())
      if (summaryRes.ok) setSummary(await summaryRes.json())
    } catch (_) {}
  }

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    let socket
    let retry
    const connect = () => {
      socket = new WebSocket(WS)
      socket.onopen = () => setConnected(true)
      socket.onclose = () => {
        setConnected(false)
        retry = setTimeout(connect, 1500)
      }
      socket.onmessage = (message) => {
        const update = JSON.parse(message.data)
        setMachines((current) => {
          const next = current.filter((m) => m.machine_id !== update.machine_id)
          next.push(update)
          const order = { critical: 0, warning: 1, healthy: 2 }
          return next.sort((a, b) => (order[a.risk_level] ?? 9) - (order[b.risk_level] ?? 9))
        })
        setSummary((current) => ({ ...current }))
        setSelected((current) => current?.machine_id === update.machine_id ? update : current)
      }
    }
    connect()
    return () => {
      clearTimeout(retry)
      socket?.close()
    }
  }, [])

  const topRisk = useMemo(
    () => [...machines].sort((a, b) => b.failure_probability - a.failure_probability).slice(0, 8),
    [machines],
  )

  const selectedMachine = selected || topRisk[0]

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">ML OPERATIONS • LIVE FACTORY</p>
          <h1>Predictive Maintenance Command Center</h1>
          <p className="subtitle">Failure risk, anomaly detection, and remaining useful life from streaming machine telemetry.</p>
        </div>
        <div className={`connection ${connected ? 'online' : ''}`}>
          <i /> {connected ? 'LIVE STREAM' : 'RECONNECTING'}
        </div>
      </header>

      <section className="metrics">
        <MetricCard label="Machines online" value={summary.machines_online ?? machines.length} detail="rolling telemetry state" />
        <MetricCard label="Healthy" value={summary.healthy ?? machines.filter(x => x.risk_level === 'healthy').length} detail="no immediate action" />
        <MetricCard label="Warning" value={summary.warning ?? machines.filter(x => x.risk_level === 'warning').length} detail="inspection recommended" />
        <MetricCard label="Critical" value={summary.critical ?? machines.filter(x => x.risk_level === 'critical').length} detail="urgent maintenance" />
      </section>

      <section className="grid">
        <article className="panel machine-table">
          <div className="panel-head">
            <div><span>Fleet risk</span><h2>Machines requiring attention</h2></div>
            <small>Sorted by failure probability</small>
          </div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Machine</th><th>Status</th><th>Failure risk</th><th>Anomaly</th><th>RUL</th></tr></thead>
              <tbody>
                {topRisk.length ? topRisk.map(machine => (
                  <tr key={machine.machine_id} onClick={() => setSelected(machine)} className={selectedMachine?.machine_id === machine.machine_id ? 'selected' : ''}>
                    <td>{machine.machine_id}</td>
                    <td><RiskBadge level={machine.risk_level} /></td>
                    <td>{pct(machine.failure_probability)}</td>
                    <td>{pct(machine.anomaly_score)}</td>
                    <td>{hours(machine.remaining_useful_life_hours)}</td>
                  </tr>
                )) : <tr><td colSpan="5" className="empty">Waiting for stream data…</td></tr>}
              </tbody>
            </table>
          </div>
        </article>

        <article className="panel focus">
          <div className="panel-head">
            <div><span>Machine detail</span><h2>{selectedMachine?.machine_id || 'No machine selected'}</h2></div>
            {selectedMachine && <RiskBadge level={selectedMachine.risk_level} />}
          </div>
          {selectedMachine ? <>
            <div className="risk-score">
              <div>
                <span>Failure probability</span>
                <strong>{pct(selectedMachine.failure_probability)}</strong>
              </div>
              <div className="bar"><i style={{ width: pct(selectedMachine.failure_probability) }} /></div>
            </div>
            <div className="signal-grid">
              <div><span>Anomaly score</span><strong>{pct(selectedMachine.anomaly_score)}</strong></div>
              <div><span>Remaining life</span><strong>{hours(selectedMachine.remaining_useful_life_hours)}</strong></div>
              <div><span>Temperature</span><strong>{selectedMachine.features?.temperature?.toFixed?.(1) ?? '—'}°</strong></div>
              <div><span>Vibration</span><strong>{selectedMachine.features?.vibration?.toFixed?.(2) ?? '—'}</strong></div>
              <div><span>RPM</span><strong>{Math.round(selectedMachine.features?.rpm || 0)}</strong></div>
              <div><span>Load</span><strong>{pct(selectedMachine.features?.load)}</strong></div>
            </div>
            <p className="timestamp">Last logical telemetry: {selectedMachine.timestamp}</p>
          </> : <p className="empty">The dashboard will populate after rolling feature windows warm up.</p>}
        </article>
      </section>

      <footer>
        <span>Kafka → online features → ML inference → PostgreSQL / Redis → WebSocket</span>
        <span>Prometheus + Grafana observability</span>
      </footer>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<App />)
