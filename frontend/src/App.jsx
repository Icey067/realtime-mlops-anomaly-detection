import React, { useState, useEffect } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine
} from 'recharts';
import {
  ShieldAlert, Activity, Cpu, Database, Zap, RefreshCw, AlertTriangle,
  Flame, Check
} from 'lucide-react';

const EMPTY_METRICS = {
  throughput: 0,
  p95Latency: null,
  avgLatency: null,
  totalScored: 0,
  anomalies: 0,
  anomalyRate: 0,
  driftScore: null,
  driftThreshold: 0.15,
  threshold: null,
  uptimeSeconds: 0,
  redisConnected: false,
  modelLoaded: false,
};

function formatUptime(seconds) {
  const total = Math.max(Math.floor(seconds || 0), 0);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

export default function App() {
  const [metrics, setMetrics] = useState(EMPTY_METRICS);
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [filter, setFilter] = useState('ALL');
  const [injecting, setInjecting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [liveResponse, healthResponse] = await Promise.all([
          fetch('/api/telemetry/live'),
          fetch('/healthz'),
        ]);
        if (!liveResponse.ok) {
          throw new Error(`Live telemetry returned ${liveResponse.status}`);
        }

        const data = await liveResponse.json();
        const health = healthResponse.ok ? await healthResponse.json() : null;
        if (cancelled) return;

        setMetrics({
          throughput: data.throughput_tx_per_sec ?? 0,
          p95Latency: data.p95_latency_ms ?? null,
          avgLatency: data.avg_latency_ms ?? null,
          totalScored: data.total_scored ?? 0,
          anomalies: data.total_anomalies ?? 0,
          anomalyRate: data.anomaly_rate_percent ?? 0,
          driftScore: typeof data.drift_score === 'number' ? data.drift_score : null,
          driftThreshold: data.drift_threshold ?? 0.15,
          threshold: typeof data.model_threshold === 'number' ? data.model_threshold : null,
          uptimeSeconds: data.uptime_seconds ?? 0,
          redisConnected: Boolean(health?.redis_connected),
          modelLoaded: Boolean(health?.model_loaded),
        });

        if (data.recent_transactions?.length > 0) {
          setTransactions(data.recent_transactions.map(t => ({
            id: t.transaction_id,
            time: new Date(t.timestamp * 1000).toLocaleTimeString(),
            userId: t.user_id,
            amount: t.amount,
            mse: t.reconstruction_mse,
            latency: t.inference_latency_ms,
            isAnomaly: t.is_anomaly,
            status: t.status,
          })));

          setLatencyHistory([...data.recent_transactions].reverse().map(t => ({
            time: new Date(t.timestamp * 1000).toLocaleTimeString(),
            latency: t.inference_latency_ms,
            mse: t.reconstruction_mse,
          })));
        }

        setError(null);
      } catch {
        if (!cancelled) {
          setMetrics(prev => ({ ...prev, redisConnected: false }));
          setError('Live data unavailable. Showing the last values received from the API.');
        }
      }
    };

    load();
    const interval = setInterval(load, 1500);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const triggerManualInjection = async (anomaly = false) => {
    setInjecting(true);
    const payload = {
      transaction_id: `tx_manual_${Date.now()}`,
      timestamp: Date.now() / 1000,
      user_id: `usr_${Math.floor(100000 + Math.random() * 900000)}`,
      amount: anomaly ? 9450.00 : 75.50,
      currency: 'USD',
      latitude: 40.7128,
      longitude: -74.0060,
      time_step: 42.0,
      v1: anomaly ? 5.8 : -0.2,
      v2: anomaly ? -4.2 : 0.1,
      v3: anomaly ? 6.1 : 0.4,
    };

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`Prediction returned ${response.status}`);
      }

      const scored = await response.json();
      const transaction = {
        id: scored.transaction_id,
        time: new Date().toLocaleTimeString(),
        userId: payload.user_id,
        amount: payload.amount,
        mse: scored.reconstruction_mse,
        latency: scored.inference_latency_ms,
        isAnomaly: scored.is_anomaly,
        status: scored.status,
      };
      setTransactions(prev => [transaction, ...prev.slice(0, 24)]);
      setLatencyHistory(prev => [
        ...prev.slice(-29),
        {
          time: transaction.time,
          latency: scored.inference_latency_ms,
          mse: scored.reconstruction_mse,
        },
      ]);
      setError(null);
    } catch {
      setError('Prediction request failed. Check that the API is running.');
    } finally {
      setTimeout(() => setInjecting(false), 300);
    }
  };

  const filteredTransactions = transactions.filter(t => {
    if (filter === 'ANOMALY') return t.isAnomaly === true;
    if (filter === 'NORMAL') return t.isAnomaly === false;
    return true;
  });

  const driftStatus = metrics.driftScore === null
    ? 'Awaiting consumer'
    : metrics.driftScore > metrics.driftThreshold
      ? 'Retrain threshold exceeded'
      : 'Below threshold';

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
            <p className="text-xs text-slate-400">Streaming Transaction Anomaly Detection</p>
          </div>
        </div>

        {/* System Badges & Controls */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Model:</span>
            <span className={`font-mono font-semibold ${metrics.modelLoaded ? 'text-cyan-300' : 'text-amber-300'}`}>
              {metrics.modelLoaded ? 'ONNX INT8' : 'Not loaded'}
            </span>
          </div>

          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-400">Redis:</span>
            <span className={`font-mono font-semibold ${metrics.redisConnected ? 'text-emerald-300' : 'text-amber-300'}`}>
              {metrics.redisConnected ? 'Connected' : 'Unavailable'}
            </span>
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
        {error && (
          <div className="rounded-lg border border-amber-500/40 bg-amber-950/30 px-4 py-3 text-xs text-amber-200">
            {error}
          </div>
        )}
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
              <span>Last 10s</span>
              <span className="font-mono text-slate-300">Uptime {formatUptime(metrics.uptimeSeconds)}</span>
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
              <span>Last {latencyHistory.length} predictions</span>
              <span className={metrics.p95Latency === null ? 'text-slate-500' : 'text-cyan-400 font-semibold'}>
                {metrics.p95Latency === null ? 'No samples' : 'Measured'}
              </span>
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

          {/* 4. Feature Drift Score */}
          <div className="glass-panel rounded-xl p-4 relative overflow-hidden group hover:border-cyan-500/30 transition">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs uppercase tracking-wider font-medium">Feature Drift Score</span>
              <RefreshCw className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold tracking-tight font-mono text-indigo-300">
                {metrics.driftScore === null ? '—' : metrics.driftScore.toFixed(3)}
              </span>
              <span className="text-xs font-medium text-slate-400">KS / Wasserstein</span>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
              <span>Alert threshold: &gt;{metrics.driftThreshold.toFixed(2)}</span>
              <span className={metrics.driftScore === null ? 'text-slate-500' : 'text-cyan-400 font-semibold'}>
                {driftStatus}
              </span>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Inference latency */}
          <div className="glass-panel rounded-xl p-5 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Real-Time Inference Latency</h3>
                <p className="text-xs text-slate-400">Quantized ONNX CPU execution time</p>
              </div>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-800 text-cyan-300">
                Avg: {metrics.avgLatency === null ? '—' : `${metrics.avgLatency}ms`}
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
                  <YAxis domain={[0, 'auto']} stroke="#475569" fontSize={11} tickFormatter={v => `${v}ms`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0c1017', borderColor: '#334155', borderRadius: 8 }}
                    itemStyle={{ color: '#00f2fe' }}
                  />
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
                Cutoff: {metrics.threshold === null ? '—' : metrics.threshold.toFixed(2)}
              </span>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={latencyHistory} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 'auto']} stroke="#475569" fontSize={11} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0c1017', borderColor: '#334155', borderRadius: 8 }}
                    itemStyle={{ color: '#a78bfa' }}
                  />
                  {metrics.threshold !== null && (
                    <ReferenceLine y={metrics.threshold} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: 'Anomaly Threshold (98%)', fill: '#f43f5e', fontSize: 10 }} />
                  )}
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
              <h3 className="text-sm font-semibold text-white">Recent API Predictions</h3>
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
                    <td className={`py-2.5 px-4 text-right font-semibold ${tx.isAnomaly === true ? 'text-rose-400' : 'text-slate-300'}`}>
                      {tx.mse === null || tx.mse === undefined ? '—' : tx.mse.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-4 text-right text-emerald-400">
                      {tx.latency === null || tx.latency === undefined ? '—' : `${tx.latency}ms`}
                    </td>
                    <td className="py-2.5 px-4 text-center">
                      {tx.isAnomaly === null || tx.isAnomaly === undefined ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          <AlertTriangle className="w-2.5 h-2.5" /> DEGRADED
                        </span>
                      ) : tx.isAnomaly ? (
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
                {filteredTransactions.length === 0 && (
                  <tr>
                    <td colSpan={7} className="py-8 px-4 text-center text-slate-500 font-sans">
                      No predictions yet. Send a request to /predict or inject a transaction.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>

      {/* Footer Status Bar */}
      <footer className="border-t border-white/10 bg-[#0c1017] px-6 py-3 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <span className={`flex items-center gap-1.5 font-medium ${metrics.redisConnected ? 'text-emerald-400' : 'text-amber-400'}`}>
            <span className={`w-2 h-2 rounded-full ${metrics.redisConnected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}></span>
            {metrics.redisConnected ? 'Redis connected' : 'Redis unavailable'}
          </span>
          <span className="text-slate-600">|</span>
          <span className={error ? 'text-amber-400' : 'text-emerald-400'}>
            {error ? 'API unavailable' : 'API connected'}
          </span>
          <span className="text-slate-600">|</span>
          <span>
            Model: <span className="text-slate-200 font-mono">
              {metrics.modelLoaded ? 'ONNX INT8 loaded' : 'degraded'}
            </span>
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span>Sentinel v1.0.0</span>
        </div>
      </footer>
    </div>
  );
}
