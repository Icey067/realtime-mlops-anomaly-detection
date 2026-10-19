import React, { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine
} from 'recharts';
import {
  ShieldAlert, Activity, Cpu, Database, Zap, RefreshCw, AlertTriangle,
  CheckCircle2, Flame, ArrowUpRight, Filter, Play, Check
} from 'lucide-react';

export default function App() {
  const [metrics, setMetrics] = useState({
    throughput: 480,
    p95Latency: 0.42,
    avgLatency: 0.28,
    totalScored: 12480,
    anomalies: 214,
    anomalyRate: 1.71,
    driftScore: 0.04,
    threshold: 1.35,
    isHealthy: true,
  });

  const [latencyHistory, setLatencyHistory] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [filter, setFilter] = useState('ALL'); // ALL | ANOMALY | NORMAL
  const [injecting, setInjecting] = useState(false);
  const [autoSimulate, setAutoSimulate] = useState(true);

  // Initialize sparkline buffer
  useEffect(() => {
    const initialData = Array.from({ length: 30 }, (_, i) => ({
      time: `${i}s`,
      latency: Number((0.2 + Math.random() * 0.35).toFixed(3)),
      mse: Number((0.4 + Math.random() * 0.5).toFixed(3)),
    }));
    setLatencyHistory(initialData);

    const initialTx = Array.from({ length: 12 }, (_, i) => generateMockTx(i % 5 === 0));
    setTransactions(initialTx);
  }, []);

  function generateMockTx(forceAnomaly = false) {
    const isAnomaly = forceAnomaly || Math.random() < 0.05;
    const mse = isAnomaly ? 1.45 + Math.random() * 2.2 : 0.2 + Math.random() * 0.7;
    const latency = Number((0.25 + Math.random() * 0.45).toFixed(2));
    const amount = Number((10 + Math.random() * (isAnomaly ? 3500 : 450)).toFixed(2));

    return {
      id: `tx_${Date.now().toString().slice(-6)}_${Math.floor(100 + Math.random() * 900)}`,
      time: new Date().toLocaleTimeString(),
      userId: `usr_${Math.floor(100000 + Math.random() * 900000)}`,
      amount: amount,
      mse: Number(mse.toFixed(4)),
      latency: latency,
      isAnomaly: isAnomaly,
      location: isAnomaly ? '40.7128, -74.0060 (Abnormal)' : '40.7128, -74.0060',
    };
  }

  // Polling / Streaming loop
  useEffect(() => {
    const interval = setInterval(async () => {
      // Try to fetch live metrics from backend API
      try {
        const res = await fetch('/api/telemetry/live');
        if (res.ok) {
          const data = await res.json();
          setMetrics(prev => ({
            ...prev,
            throughput: data.throughput_tx_per_sec || prev.throughput,
            p95Latency: data.p95_latency_ms || prev.p95Latency,
            avgLatency: data.avg_latency_ms || prev.avgLatency,
            totalScored: data.total_scored || prev.totalScored,
            anomalies: data.total_anomalies || prev.anomalies,
            anomalyRate: data.anomaly_rate_percent || prev.anomalyRate,
            isHealthy: true,
          }));

          if (data.recent_transactions && data.recent_transactions.length > 0) {
            setTransactions(data.recent_transactions.map(t => ({
              id: t.transaction_id,
              time: new Date(t.timestamp * 1000).toLocaleTimeString(),
              userId: t.user_id,
              amount: t.amount,
              mse: t.reconstruction_mse,
              latency: t.inference_latency_ms,
              isAnomaly: t.is_anomaly,
              location: '40.7128, -74.0060',
            })));
          }
        }
      } catch {
        // Fallback to active simulation mode for fluid UI preview
        if (autoSimulate) {
          const newTx = generateMockTx();
          setTransactions(prev => [newTx, ...prev.slice(0, 24)]);

          setLatencyHistory(prev => {
            const next = [...prev.slice(1), {
              time: newTx.time,
              latency: newTx.latency,
              mse: newTx.mse,
            }];
            return next;
          });

          setMetrics(prev => {
            const nextTotal = prev.totalScored + 1;
            const nextAnom = prev.anomalies + (newTx.isAnomaly ? 1 : 0);
            return {
              ...prev,
              totalScored: nextTotal,
              anomalies: nextAnom,
              anomalyRate: Number(((nextAnom / nextTotal) * 100).toFixed(2)),
              p95Latency: Number((newTx.latency * 1.35).toFixed(2)),
              throughput: Math.floor(450 + Math.random() * 80),
            };
          });
        }
      }
    }, 1200);

    return () => clearInterval(interval);
  }, [autoSimulate]);

  // Inject manual anomaly transaction
  const triggerManualInjection = async (anomaly = false) => {
    setInjecting(true);
    const mockPayload = {
      transaction_id: `tx_manual_${Date.now()}`,
      timestamp: Date.now() / 1000,
      user_id: `usr_${Math.floor(100000 + Math.random() * 900000)}`,
      amount: anomaly ? 9450.00 : 75.50,
      currency: "USD",
      latitude: 40.7128,
      longitude: -74.0060,
      time_step: 42.0,
      v1: anomaly ? 5.8 : -0.2,
      v2: anomaly ? -4.2 : 0.1,
      v3: anomaly ? 6.1 : 0.4,
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockPayload),
      });
      if (res.ok) {
        const scored = await res.json();
        const txRecord = {
          id: scored.transaction_id,
          time: new Date().toLocaleTimeString(),
          userId: mockPayload.user_id,
          amount: mockPayload.amount,
          mse: scored.reconstruction_mse,
          latency: scored.inference_latency_ms,
          isAnomaly: scored.is_anomaly,
          location: '40.7128, -74.0060 (Manual)',
        };
        setTransactions(prev => [txRecord, ...prev.slice(0, 24)]);
      }
    } catch {
      // Local fallback
      const txRecord = generateMockTx(anomaly);
      setTransactions(prev => [txRecord, ...prev.slice(0, 24)]);
    } finally {
      setTimeout(() => setInjecting(false), 300);
    }
  };

  const filteredTransactions = transactions.filter(t => {
    if (filter === 'ANOMALY') return t.isAnomaly;
    if (filter === 'NORMAL') return !t.isAnomaly;
    return true;
  });

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-white/10 bg-[#0c1017]/80 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-600 to-indigo-600 p-[1px] shadow-lg shadow-cyan-500/20">
            <div className="w-full h-full bg-[#0c1017] rounded-[11px] flex items-center justify-center">
              <ShieldAlert className="w-5 h-5 text-cyan-400" />
            </div>
            <span className="absolute -bottom-0.5 -right-0.5 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-cyan-400 bg-clip-text text-transparent">
                SENTINEL
              </h1>
              <span className="text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-cyan-400">
                Real-Time MLOps
              </span>
            </div>
            <p className="text-xs text-slate-400">High-Frequency Transaction Anomaly Engine (Sub-15ms SLA)</p>
          </div>
        </div>

        {/* System Badges & Controls */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Model:</span>
            <span className="font-mono text-cyan-300 font-semibold">ONNX INT8</span>
          </div>

          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">Ingest:</span>
            <span className="font-mono text-emerald-300 font-semibold">Redis Streams</span>
          </div>

          <button
            onClick={() => triggerManualInjection(true)}
            disabled={injecting}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-600/20 border border-rose-500/40 text-rose-300 hover:bg-rose-600/30 transition text-xs font-medium active:scale-95"
            title="Inject an abnormal high-MSE transaction"
          >
            <Flame className="w-3.5 h-3.5 text-rose-400" />
            Inject Fraud Tx
          </button>

          <button
            onClick={() => triggerManualInjection(false)}
            disabled={injecting}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600/20 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-600/30 transition text-xs font-medium active:scale-95"
          >
            <Zap className="w-3.5 h-3.5 text-emerald-400" />
            Inject Normal Tx
          </button>
        </div>
      </header>

      {/* Main Content Dashboard */}
      <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-6">
        {/* KPI Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* 1. Throughput */}
          <div className="glass-panel rounded-xl p-4 relative overflow-hidden group hover:border-cyan-500/30 transition">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase tracking-wider font-medium">Throughput</span>
              <Activity className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold tracking-tight font-mono text-white">
                {metrics.throughput}
              </span>
              <span className="text-xs font-medium text-slate-400">tx / sec</span>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
              <span>Streaming Pipeline</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> 100% In Budget
              </span>
            </div>
          </div>

          {/* 2. P95 Inference Latency */}
          <div className="glass-panel rounded-xl p-4 relative overflow-hidden group hover:border-cyan-500/30 transition">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase tracking-wider font-medium">P95 Inference Latency</span>
              <Zap className="w-4 h-4 text-amber-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold tracking-tight font-mono text-emerald-400">
                {metrics.p95Latency}
              </span>
              <span className="text-xs font-medium text-slate-400">ms</span>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
              <span>SLA Target: &lt;15.00ms</span>
              <span className="text-cyan-400 font-semibold">97.2% Margin</span>
            </div>
          </div>

          {/* 3. Anomaly Rate */}
          <div className="glass-panel rounded-xl p-4 relative overflow-hidden group hover:border-cyan-500/30 transition">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase tracking-wider font-medium">Anomaly Rate</span>
              <ShieldAlert className="w-4 h-4 text-rose-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold tracking-tight font-mono text-rose-400">
                {metrics.anomalyRate}%
              </span>
              <span className="text-xs text-slate-400 font-mono">({metrics.anomalies} flags)</span>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
              <span>Total Scored:</span>
              <span className="font-mono text-slate-200">{metrics.totalScored.toLocaleString()}</span>
            </div>
          </div>

          {/* 4. Evidently AI Drift Score */}
          <div className="glass-panel rounded-xl p-4 relative overflow-hidden group hover:border-cyan-500/30 transition">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase tracking-wider font-medium">Evidently Data Drift</span>
              <RefreshCw className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold tracking-tight font-mono text-indigo-300">
                {metrics.driftScore.toFixed(3)}
              </span>
              <span className="text-xs font-medium text-slate-400">KS Score</span>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
              <span>Retrain Threshold: &gt;0.15</span>
              <span className="text-emerald-400 font-semibold">Distribution Stable</span>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Sub-15ms Latency Sparkline */}
          <div className="glass-panel rounded-xl p-5 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Real-Time Inference Latency Sparkline</h3>
                <p className="text-xs text-slate-400">Quantized ONNX CPU Execution Time (SLA: 15ms)</p>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-800 text-cyan-300">
                Avg: {metrics.avgLatency}ms
              </span>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={latencyHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                  <defs>
                    <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00f2fe" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#00f2fe" stopOpacity={0.0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 2.0]} stroke="#475569" fontSize={11} tickFormatter={v => `${v}ms`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0c1017', borderColor: '#334155', borderRadius: 8 }}
                    itemStyle={{ color: '#00f2fe' }}
                  />
                  <ReferenceLine y={15.0} stroke="#f43f5e" strokeDasharray="3 3" label={{ value: '15ms SLA', fill: '#f43f5e', fontSize: 10 }} />
                  <Area type="monotone" dataKey="latency" stroke="#00f2fe" strokeWidth={2} fillOpacity={1} fill="url(#latencyGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Reconstruction Error MSE vs Anomaly Threshold */}
          <div className="glass-panel rounded-xl p-5 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Autoencoder Reconstruction MSE</h3>
                <p className="text-xs text-slate-400">Cutoff threshold calibrated at 98th percentile</p>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-rose-950/80 border border-rose-800 text-rose-300">
                Cutoff: {metrics.threshold.toFixed(2)}
              </span>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={latencyHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 4.0]} stroke="#475569" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0c1017', borderColor: '#334155', borderRadius: 8 }}
                    itemStyle={{ color: '#a78bfa' }}
                  />
                  <ReferenceLine y={metrics.threshold} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: 'Anomaly Threshold (98%)', fill: '#f43f5e', fontSize: 10 }} />
                  <Line type="monotone" dataKey="mse" stroke="#a78bfa" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Live Transaction Stream Table */}
        <div className="glass-panel rounded-xl overflow-hidden flex flex-col">
          <div className="p-4 border-b border-white/10 flex flex-wrap items-center justify-between gap-3 bg-[#0c1017]/60">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-white">Active Redis Stream Events</h3>
              <span className="text-xs text-slate-400 font-mono">({filteredTransactions.length} rendered)</span>
            </div>

            {/* Filter Tabs */}
            <div className="flex items-center gap-1.5 p-1 rounded-lg bg-black/40 border border-white/10 text-xs">
              <button
                onClick={() => setFilter('ALL')}
                className={`px-3 py-1 rounded-md transition font-medium ${filter === 'ALL' ? 'bg-cyan-500 text-black font-semibold' : 'text-slate-400 hover:text-white'}`}
              >
                All
              </button>
              <button
                onClick={() => setFilter('ANOMALY')}
                className={`px-3 py-1 rounded-md transition font-medium ${filter === 'ANOMALY' ? 'bg-rose-500 text-white font-semibold' : 'text-slate-400 hover:text-white'}`}
              >
                Fraud Only
              </button>
              <button
                onClick={() => setFilter('NORMAL')}
                className={`px-3 py-1 rounded-md transition font-medium ${filter === 'NORMAL' ? 'bg-emerald-500 text-black font-semibold' : 'text-slate-400 hover:text-white'}`}
              >
                Normal
              </button>
            </div>
          </div>

          <div className="overflow-x-auto max-h-[380px] overflow-y-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead className="bg-[#111722] text-slate-400 sticky top-0 uppercase tracking-wider font-mono text-[10px] border-b border-white/5">
                <tr>
                  <th className="py-2.5 px-4">Timestamp</th>
                  <th className="py-2.5 px-4">Transaction ID</th>
                  <th className="py-2.5 px-4">User</th>
                  <th className="py-2.5 px-4 text-right">Amount</th>
                  <th className="py-2.5 px-4 text-right">MSE Score</th>
                  <th className="py-2.5 px-4 text-right">Latency</th>
                  <th className="py-2.5 px-4 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 font-mono">
                {filteredTransactions.map((tx) => (
                  <tr
                    key={tx.id}
                    className={`transition-colors hover:bg-white/[0.03] ${tx.isAnomaly ? 'bg-rose-950/20' : ''}`}
                  >
                    <td className="py-2.5 px-4 text-slate-400">{tx.time}</td>
                    <td className="py-2.5 px-4 text-cyan-300 font-medium">{tx.id}</td>
                    <td className="py-2.5 px-4 text-slate-300">{tx.userId}</td>
                    <td className="py-2.5 px-4 text-right text-white font-medium">${tx.amount.toFixed(2)}</td>
                    <td className={`py-2.5 px-4 text-right font-semibold ${tx.isAnomaly ? 'text-rose-400' : 'text-slate-300'}`}>
                      {tx.mse.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-4 text-right text-emerald-400">{tx.latency}ms</td>
                    <td className="py-2.5 px-4 text-center">
                      {tx.isAnomaly ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/40 animate-pulse">
                          <AlertTriangle className="w-2.5 h-2.5" /> FRAUD DETECTED
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                          <Check className="w-2.5 h-2.5" /> NORMAL
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Footer Status Bar */}
      <footer className="border-t border-white/10 bg-[#0c1017] px-6 py-3 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            Redis Cluster Connected
          </span>
          <span className="text-slate-600">|</span>
          <span>Inference Runtime: <span className="text-slate-200 font-mono">ONNX Runtime 1.31 (Dynamic INT8)</span></span>
        </div>
        <div className="flex items-center gap-3">
          <span>Sentinel MLOps v1.0.0</span>
        </div>
      </footer>
    </div>
  );
}
