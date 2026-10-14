import React, { useState } from 'react';
import {
  FileText,
  Download,
  Copy,
  Trash2,
  Undo2,
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Bot,
  User,
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
} from 'lucide-react';

export interface LoggedTurn {
  turn_idx: number;
  timestamp: string;
  user_text: string;
  source_weights: {
    valence: { w_prior: number; w_audio: number; w_visual: number; w_sensory: number };
    arousal: { w_prior: number; w_audio: number; w_visual: number; w_sensory: number };
  };
  likelihood: {
    valence: { mu_sensory: number; var_sensory: number; mu_audio: number; var_audio: number; mu_visual: number; var_visual: number };
    arousal: { mu_sensory: number; var_sensory: number; mu_audio: number; var_audio: number; mu_visual: number; var_visual: number };
  };
  prior: {
    valence: { name: string; mu: number; var: number };
    arousal: { name: string; mu: number; var: number };
  };
  posterior: {
    valence: { mu: number; var: number; sigma: number };
    arousal: { mu: number; var: number; sigma: number };
  };
  gating_and_policy: {
    delta_valence: number;
    delta_arousal: number;
    delta_max: number;
    sigma_valence: number;
    sigma_arousal: number;
    sigma_max: number;
    action: string;
    label: string;
    triggers: string[];
    context_warmup: boolean;
  };
  assistant_text: string;
  gesture: string;
  latency_ms: number;
  timing_breakdown_ms?: { [key: string]: number };
}

interface Props {
  logs: LoggedTurn[];
  enableLogging: boolean;
  onToggleLogging: (enabled: boolean) => void;
  onClearLogs: () => void;
  onDeleteLastTurn?: () => void;
}

export const InteractionLogTable: React.FC<Props> = ({
  logs,
  enableLogging,
  onToggleLogging,
  onClearLogs,
  onDeleteLastTurn,
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(true);
  const [dimensionFilter, setDimensionFilter] = useState<'both' | 'valence' | 'arousal'>('both');
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  // Download CSV
  const handleDownloadCSV = () => {
    if (logs.length === 0) return;

    const headers = [
      'Turn #',
      'Timestamp',
      'User Utterance',
      'Valence w_C',
      'Valence w_A',
      'Valence w_V',
      'Valence w_AV',
      'Valence Sensory Likelihood (mu, var)',
      'Valence Prior (mu, var)',
      'Valence Posterior (mu, var)',
      'Arousal w_C',
      'Arousal w_A',
      'Arousal w_V',
      'Arousal w_AV',
      'Arousal Sensory Likelihood (mu, var)',
      'Arousal Prior (mu, var)',
      'Arousal Posterior (mu, var)',
      'Delta* Valence',
      'Delta* Arousal',
      'Delta* Max',
      'Sigma_post Max',
      'Policy Action',
      'Context Warmup',
      'Robot Response',
      'Gesture',
      'Latency (ms)',
    ];

    const rows = logs.map((log) => [
      log.turn_idx,
      `"${log.timestamp}"`,
      `"${log.user_text.replace(/"/g, '""')}"`,
      log.source_weights.valence.w_prior,
      log.source_weights.valence.w_audio,
      log.source_weights.valence.w_visual,
      log.source_weights.valence.w_sensory,
      `"N(${log.likelihood.valence.mu_sensory.toFixed(2)}, ${log.likelihood.valence.var_sensory.toFixed(3)})"`,
      `"N(${log.prior.valence.mu.toFixed(2)}, ${log.prior.valence.var.toFixed(3)})"`,
      `"N(${log.posterior.valence.mu.toFixed(2)}, ${log.posterior.valence.var.toFixed(3)})"`,
      log.source_weights.arousal.w_prior,
      log.source_weights.arousal.w_audio,
      log.source_weights.arousal.w_visual,
      log.source_weights.arousal.w_sensory,
      `"N(${log.likelihood.arousal.mu_sensory.toFixed(2)}, ${log.likelihood.arousal.var_sensory.toFixed(3)})"`,
      `"N(${log.prior.arousal.mu.toFixed(2)}, ${log.prior.arousal.var.toFixed(3)})"`,
      `"N(${log.posterior.arousal.mu.toFixed(2)}, ${log.posterior.arousal.var.toFixed(3)})"`,
      log.gating_and_policy.delta_valence,
      log.gating_and_policy.delta_arousal,
      log.gating_and_policy.delta_max,
      log.gating_and_policy.sigma_max,
      `"${log.gating_and_policy.action}"`,
      log.gating_and_policy.context_warmup ? 'Yes' : 'No',
      `"${log.assistant_text.replace(/"/g, '""')}"`,
      `"${log.gesture}"`,
      log.latency_ms,
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `live_interaction_sample_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Download JSON
  const handleDownloadJSON = () => {
    if (logs.length === 0) return;
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `live_interaction_sample_${new Date().toISOString().slice(0, 10)}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Copy LaTeX Table formatted for IEEE / ACM paper
  const handleCopyLaTeX = () => {
    if (logs.length === 0) return;

    let latex = `% Sample Multimodal Human-Robot Interaction Flow\n`;
    latex += `\\begin{table*}[t]\n\\centering\n\\small\n`;
    latex += `\\caption{Sample Multimodal Interaction Flow with Turn-Level Bayesian Updates and Policy Gating}\n`;
    latex += `\\label{tab:sample_interaction_flow}\n`;
    latex += `\\begin{tabular}{c p{3.6cm} c c c c c c c}\n`;
    latex += `\\hline\n`;
    latex += `Turn & User Utterance & Dim & Likelihood $\\mathcal{N}(\\mu, \\sigma^2)$ & Prior $\\mathcal{N}(\\mu, \\sigma^2)$ & Posterior $\\mathcal{N}(\\mu, \\sigma^2)$ & Weights $(w_C, w_A, w_V)$ & Policy (\\& Gesture) & Latency \\\\\n`;
    latex += `\\hline\n`;

    logs.forEach((log) => {
      const v = log;
      const vLik = `$\\mathcal{N}(${v.likelihood.valence.mu_sensory.toFixed(2)}, ${v.likelihood.valence.var_sensory.toFixed(2)})$`;
      const vPri = `$\\mathcal{N}(${v.prior.valence.mu.toFixed(2)}, ${v.prior.valence.var.toFixed(2)})$`;
      const vPos = `$\\mathcal{N}(${v.posterior.valence.mu.toFixed(2)}, ${v.posterior.valence.var.toFixed(2)})$`;
      const vW = `(${v.source_weights.valence.w_prior.toFixed(2)}, ${v.source_weights.valence.w_audio.toFixed(2)}, ${v.source_weights.valence.w_visual.toFixed(2)})`;

      const aLik = `$\\mathcal{N}(${v.likelihood.arousal.mu_sensory.toFixed(2)}, ${v.likelihood.arousal.var_sensory.toFixed(2)})$`;
      const aPri = `$\\mathcal{N}(${v.prior.arousal.mu.toFixed(2)}, ${v.prior.arousal.var.toFixed(2)})$`;
      const aPos = `$\\mathcal{N}(${v.posterior.arousal.mu.toFixed(2)}, ${v.posterior.arousal.var.toFixed(2)})$`;
      const aW = `(${v.source_weights.arousal.w_prior.toFixed(2)}, ${v.source_weights.arousal.w_audio.toFixed(2)}, ${v.source_weights.arousal.w_visual.toFixed(2)})`;

      const cleanText = v.user_text.replace(/&/g, '\\&').replace(/%/g, '\\%');
      const actionText = `\\texttt{${v.gating_and_policy.action}} (\\texttt{${v.gesture}})`;
      const latText = `${Math.round(v.latency_ms)}\\,ms`;

      latex += `\\multirow{2}{*}{${v.turn_idx}} & \\multirow{2}{=}{\`\`${cleanText}''} & V & ${vLik} & ${vPri} & ${vPos} & ${vW} & \\multirow{2}{*}{${actionText}} & \\multirow{2}{*}{${latText}} \\\\\n`;
      latex += ` & & A & ${aLik} & ${aPri} & ${aPos} & ${aW} & & \\\\\n`;
      latex += `\\hline\n`;
    });

    latex += `\\end{tabular}\n\\end{table*}\n`;

    navigator.clipboard.writeText(latex).then(() => {
      setCopyStatus('LaTeX Copied!');
      setTimeout(() => setCopyStatus(null), 2500);
    });
  };

  const getPolicyBadgeStyle = (action: string) => {
    switch (action.toLowerCase()) {
      case 'clarify':
        return {
          bg: 'rgba(244, 63, 94, 0.15)',
          color: '#fb7185',
          border: '1px solid rgba(244, 63, 94, 0.35)',
          icon: <HelpCircle size={12} color="#fb7185" />,
        };
      case 'cautious':
        return {
          bg: 'rgba(245, 158, 11, 0.15)',
          color: '#fbbf24',
          border: '1px solid rgba(245, 158, 11, 0.35)',
          icon: <AlertTriangle size={12} color="#fbbf24" />,
        };
      default:
        return {
          bg: 'rgba(16, 185, 129, 0.15)',
          color: '#34d399',
          border: '1px solid rgba(16, 185, 129, 0.35)',
          icon: <ShieldCheck size={12} color="#34d399" />,
        };
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1rem 1.25rem', marginTop: '1.25rem', overflow: 'hidden' }}>
      {/* Header bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: isExpanded ? '1rem' : 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: '0.5rem',
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(59, 130, 246, 0.25) 100%)',
              border: '1px solid rgba(99, 102, 241, 0.35)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <FileText size={17} color="#a5b4fc" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.01em' }}>
                Interaction Flow Log
              </h3>
              <span
                style={{
                  fontSize: '0.7rem',
                  padding: '0.1rem 0.45rem',
                  borderRadius: '9999px',
                  background: 'rgba(255, 255, 255, 0.08)',
                  color: '#93c5fd',
                  fontWeight: 700,
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {logs.length} {logs.length === 1 ? 'Turn' : 'Turns'} Recorded
              </span>
            </div>
            <p style={{ fontSize: '0.72rem', color: '#9ca3af' }}>
              Full Bayesian state, source weights, gate thresholds (δ*, σ_post), and end-to-end latency
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          {/* Toggle Logging Button */}
          <button
            type="button"
            onClick={() => onToggleLogging(!enableLogging)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              padding: '0.32rem 0.7rem',
              borderRadius: '0.4rem',
              background: enableLogging ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)',
              border: enableLogging ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid var(--border-subtle)',
              color: enableLogging ? '#34d399' : '#9ca3af',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: enableLogging ? '#10b981' : '#6b7280',
                boxShadow: enableLogging ? '0 0 8px #10b981' : 'none',
              }}
            />
            {enableLogging ? 'Logging: ACTIVE' : 'Logging: PAUSED'}
          </button>

          {/* Dimension Filter */}
          <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.04)', borderRadius: '0.4rem', border: '1px solid var(--border-subtle)', padding: '2px' }}>
            <button
              type="button"
              onClick={() => setDimensionFilter('both')}
              style={{
                padding: '0.2rem 0.5rem',
                fontSize: '0.72rem',
                fontWeight: 600,
                borderRadius: '0.25rem',
                border: 'none',
                background: dimensionFilter === 'both' ? 'rgba(99, 102, 241, 0.4)' : 'transparent',
                color: dimensionFilter === 'both' ? '#ffffff' : '#9ca3af',
                cursor: 'pointer',
              }}
            >
              Both (V & A)
            </button>
            <button
              type="button"
              onClick={() => setDimensionFilter('valence')}
              style={{
                padding: '0.2rem 0.5rem',
                fontSize: '0.72rem',
                fontWeight: 600,
                borderRadius: '0.25rem',
                border: 'none',
                background: dimensionFilter === 'valence' ? 'rgba(99, 102, 241, 0.4)' : 'transparent',
                color: dimensionFilter === 'valence' ? '#ffffff' : '#9ca3af',
                cursor: 'pointer',
              }}
            >
              Valence
            </button>
            <button
              type="button"
              onClick={() => setDimensionFilter('arousal')}
              style={{
                padding: '0.2rem 0.5rem',
                fontSize: '0.72rem',
                fontWeight: 600,
                borderRadius: '0.25rem',
                border: 'none',
                background: dimensionFilter === 'arousal' ? 'rgba(99, 102, 241, 0.4)' : 'transparent',
                color: dimensionFilter === 'arousal' ? '#ffffff' : '#9ca3af',
                cursor: 'pointer',
              }}
            >
              Arousal
            </button>
          </div>

          {/* CSV Export */}
          <button
            type="button"
            disabled={logs.length === 0}
            onClick={handleDownloadCSV}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.32rem 0.65rem',
              borderRadius: '0.4rem',
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-subtle)',
              color: logs.length > 0 ? '#d1d5db' : '#6b7280',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: logs.length > 0 ? 'pointer' : 'not-allowed',
            }}
            title="Download CSV"
          >
            <Download size={12} />
            CSV
          </button>

          {/* JSON Export */}
          <button
            type="button"
            disabled={logs.length === 0}
            onClick={handleDownloadJSON}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.32rem 0.65rem',
              borderRadius: '0.4rem',
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-subtle)',
              color: logs.length > 0 ? '#d1d5db' : '#6b7280',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: logs.length > 0 ? 'pointer' : 'not-allowed',
            }}
            title="Download JSON"
          >
            JSON
          </button>

          {/* LaTeX Export */}
          <button
            type="button"
            disabled={logs.length === 0}
            onClick={handleCopyLaTeX}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.32rem 0.65rem',
              borderRadius: '0.4rem',
              background: copyStatus ? 'rgba(16, 185, 129, 0.2)' : 'rgba(99, 102, 241, 0.15)',
              border: copyStatus ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(99, 102, 241, 0.35)',
              color: copyStatus ? '#34d399' : '#a5b4fc',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: logs.length > 0 ? 'pointer' : 'not-allowed',
            }}
            title="Copy LaTeX Table for Paper"
          >
            {copyStatus ? <Check size={12} color="#34d399" /> : <Copy size={12} />}
            {copyStatus || 'LaTeX Table'}
          </button>

          {/* Delete Last Turn (Undo) */}
          {onDeleteLastTurn && (
            <button
              type="button"
              disabled={logs.length === 0}
              onClick={onDeleteLastTurn}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.32rem 0.65rem',
                borderRadius: '0.4rem',
                background: logs.length > 0 ? 'rgba(245, 158, 11, 0.12)' : 'rgba(255, 255, 255, 0.05)',
                border: logs.length > 0 ? '1px solid rgba(245, 158, 11, 0.35)' : '1px solid var(--border-subtle)',
                color: logs.length > 0 ? '#fbbf24' : '#6b7280',
                fontSize: '0.75rem',
                fontWeight: 700,
                cursor: logs.length > 0 ? 'pointer' : 'not-allowed',
                transition: 'all 0.15s ease',
              }}
              title="Delete the last recorded turn and roll back conversation history"
            >
              <Undo2 size={12} />
              Undo Last Turn
            </button>
          )}

          {/* Clear Logs */}
          <button
            type="button"
            disabled={logs.length === 0}
            onClick={onClearLogs}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.32rem 0.55rem',
              borderRadius: '0.4rem',
              background: 'rgba(244, 63, 94, 0.08)',
              border: '1px solid rgba(244, 63, 94, 0.25)',
              color: '#fb7185',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: logs.length > 0 ? 'pointer' : 'not-allowed',
            }}
            title="Clear Logged Turns"
          >
            <Trash2 size={12} />
          </button>

          {/* Collapse/Expand Toggle */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              background: 'none',
              border: 'none',
              color: '#9ca3af',
              cursor: 'pointer',
              padding: '0.2rem',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
      </div>

      {/* Table Content */}
      {isExpanded && (
        <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: '0.6rem' }}>
          {logs.length === 0 ? (
            <div style={{ padding: '2.5rem', textAlign: 'center', color: '#9ca3af', fontSize: '0.85rem' }}>
              <Clock size={24} style={{ margin: '0 auto 0.5rem auto', opacity: 0.5 }} />
              No interaction turns logged yet. Turn logging is <strong>{enableLogging ? 'enabled' : 'paused'}</strong>.
              <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
                Record a turn via the webcam/microphone to begin logging the interaction flow.
              </div>
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: 'rgba(255, 255, 255, 0.03)', borderBottom: '1px solid var(--border-subtle)', color: '#d1d5db' }}>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, width: '45px' }}>#</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, minWidth: '180px' }}>User Utterance</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, minWidth: '150px' }}>
                    Weights <span style={{ color: '#ef4444' }}>w_C</span> / <span style={{ color: '#3b82f6' }}>w_A</span> / <span style={{ color: '#f59e0b' }}>w_V</span>
                  </th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, minWidth: '120px' }}>Sensory AV (μ, σ²)</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, minWidth: '120px' }}>Prior (μ, σ²)</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, minWidth: '120px' }}>Posterior (μ, σ²)</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, minWidth: '140px' }}>Gating & Policy</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, minWidth: '190px' }}>Generated Response</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, width: '90px' }}>Gesture</th>
                  <th style={{ padding: '0.6rem 0.75rem', fontWeight: 700, width: '85px' }}>Latency</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => {
                  const policyStyle = getPolicyBadgeStyle(log.gating_and_policy.action);
                  const showV = dimensionFilter === 'both' || dimensionFilter === 'valence';
                  const showA = dimensionFilter === 'both' || dimensionFilter === 'arousal';

                  return (
                    <tr
                      key={log.turn_idx}
                      style={{
                        borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                        background: log.turn_idx % 2 === 0 ? 'rgba(255, 255, 255, 0.01)' : 'transparent',
                        verticalAlign: 'top',
                      }}
                    >
                      {/* Turn # */}
                      <td style={{ padding: '0.65rem 0.75rem', fontWeight: 800, color: '#a5b4fc', fontFamily: 'var(--font-mono)' }}>
                        #{log.turn_idx}
                      </td>

                      {/* User Utterance */}
                      <td style={{ padding: '0.65rem 0.75rem', color: '#f3f4f6' }}>
                        <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'flex-start' }}>
                          <User size={13} color="#9ca3af" style={{ flexShrink: 0, marginTop: '2px' }} />
                          <span style={{ fontWeight: 500 }}>"{log.user_text}"</span>
                        </div>
                      </td>

                      {/* Source Weights */}
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'var(--font-mono)' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                          {showV && (
                            <div>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>V:</span>
                              <span style={{ color: '#fca5a5', fontWeight: 700 }}>{log.source_weights.valence.w_prior.toFixed(2)}</span>
                              <span style={{ color: '#6b7280' }}> | </span>
                              <span style={{ color: '#93c5fd', fontWeight: 700 }}>{log.source_weights.valence.w_audio.toFixed(2)}</span>
                              <span style={{ color: '#6b7280' }}> | </span>
                              <span style={{ color: '#fde68a', fontWeight: 700 }}>{log.source_weights.valence.w_visual.toFixed(2)}</span>
                            </div>
                          )}
                          {showA && (
                            <div>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>A:</span>
                              <span style={{ color: '#fca5a5', fontWeight: 700 }}>{log.source_weights.arousal.w_prior.toFixed(2)}</span>
                              <span style={{ color: '#6b7280' }}> | </span>
                              <span style={{ color: '#93c5fd', fontWeight: 700 }}>{log.source_weights.arousal.w_audio.toFixed(2)}</span>
                              <span style={{ color: '#6b7280' }}> | </span>
                              <span style={{ color: '#fde68a', fontWeight: 700 }}>{log.source_weights.arousal.w_visual.toFixed(2)}</span>
                            </div>
                          )}
                        </div>
                      </td>

                      {/* Sensory Likelihood (mu, var) */}
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'var(--font-mono)' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                          {showV && (
                            <div style={{ color: '#6ee7b7' }}>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>V:</span>
                              {log.likelihood.valence.mu_sensory >= 0 ? '+' : ''}{log.likelihood.valence.mu_sensory.toFixed(2)}
                              <span style={{ color: '#6b7280', fontSize: '0.68rem' }}> (σ²={log.likelihood.valence.var_sensory.toFixed(2)})</span>
                            </div>
                          )}
                          {showA && (
                            <div style={{ color: '#6ee7b7' }}>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>A:</span>
                              {log.likelihood.arousal.mu_sensory >= 0 ? '+' : ''}{log.likelihood.arousal.mu_sensory.toFixed(2)}
                              <span style={{ color: '#6b7280', fontSize: '0.68rem' }}> (σ²={log.likelihood.arousal.var_sensory.toFixed(2)})</span>
                            </div>
                          )}
                        </div>
                      </td>

                      {/* Prior (mu, var) */}
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'var(--font-mono)' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                          {showV && (
                            <div style={{ color: '#fca5a5' }}>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>V:</span>
                              {log.prior.valence.mu >= 0 ? '+' : ''}{log.prior.valence.mu.toFixed(2)}
                              <span style={{ color: '#6b7280', fontSize: '0.68rem' }}> (σ²={log.prior.valence.var.toFixed(2)})</span>
                            </div>
                          )}
                          {showA && (
                            <div style={{ color: '#fca5a5' }}>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>A:</span>
                              {log.prior.arousal.mu >= 0 ? '+' : ''}{log.prior.arousal.mu.toFixed(2)}
                              <span style={{ color: '#6b7280', fontSize: '0.68rem' }}> (σ²={log.prior.arousal.var.toFixed(2)})</span>
                            </div>
                          )}
                        </div>
                      </td>

                      {/* Posterior (mu, var) */}
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'var(--font-mono)' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                          {showV && (
                            <div style={{ color: '#ffffff', fontWeight: 700 }}>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>V:</span>
                              {log.posterior.valence.mu >= 0 ? '+' : ''}{log.posterior.valence.mu.toFixed(2)}
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', fontWeight: 400 }}> (σ²={log.posterior.valence.var.toFixed(2)})</span>
                            </div>
                          )}
                          {showA && (
                            <div style={{ color: '#ffffff', fontWeight: 700 }}>
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', marginRight: '4px' }}>A:</span>
                              {log.posterior.arousal.mu >= 0 ? '+' : ''}{log.posterior.arousal.mu.toFixed(2)}
                              <span style={{ color: '#9ca3af', fontSize: '0.68rem', fontWeight: 400 }}> (σ²={log.posterior.arousal.var.toFixed(2)})</span>
                            </div>
                          )}
                        </div>
                      </td>

                      {/* Gating & Policy */}
                      <td style={{ padding: '0.65rem 0.75rem' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                          {/* Policy Action Badge */}
                          <div
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.3rem',
                              padding: '0.15rem 0.45rem',
                              borderRadius: '0.35rem',
                              background: policyStyle.bg,
                              border: policyStyle.border,
                              color: policyStyle.color,
                              fontWeight: 700,
                              fontSize: '0.72rem',
                              width: 'fit-content',
                            }}
                          >
                            {policyStyle.icon}
                            {log.gating_and_policy.action.toUpperCase()}
                            {log.gating_and_policy.context_warmup && (
                              <span style={{ fontSize: '0.65rem', opacity: 0.8 }}>(warmup)</span>
                            )}
                          </div>
                          {/* Gate metrics */}
                          <div style={{ fontSize: '0.68rem', color: '#9ca3af', fontFamily: 'var(--font-mono)' }}>
                            δ*={log.gating_and_policy.delta_max.toFixed(2)} | σ={log.gating_and_policy.sigma_max.toFixed(2)}
                          </div>
                        </div>
                      </td>

                      {/* Generated Response */}
                      <td style={{ padding: '0.65rem 0.75rem', color: '#e5e7eb' }}>
                        <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'flex-start' }}>
                          <Bot size={13} color="var(--accent-primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
                          <span style={{ fontSize: '0.74rem' }}>{log.assistant_text}</span>
                        </div>
                      </td>

                      {/* Gesture */}
                      <td style={{ padding: '0.65rem 0.75rem' }}>
                        <span
                          style={{
                            padding: '0.15rem 0.45rem',
                            borderRadius: '0.3rem',
                            background: 'rgba(99, 102, 241, 0.15)',
                            border: '1px solid rgba(99, 102, 241, 0.35)',
                            color: '#a5b4fc',
                            fontWeight: 700,
                            fontSize: '0.7rem',
                            fontFamily: 'var(--font-mono)',
                          }}
                        >
                          {log.gesture}
                        </span>
                      </td>

                      {/* Latency */}
                      <td style={{ padding: '0.65rem 0.75rem', fontFamily: 'var(--font-mono)' }}>
                        <div style={{ fontWeight: 700, color: '#34d399' }}>
                          {Math.round(log.latency_ms)} ms
                        </div>
                        {log.timing_breakdown_ms && (
                          <div style={{ fontSize: '0.65rem', color: '#6b7280' }}>
                            STT:{Math.round(log.timing_breakdown_ms.stt || 0)}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
};
