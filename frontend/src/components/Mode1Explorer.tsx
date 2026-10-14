import React, { useState, useEffect, useMemo } from 'react';
import {
  calculateHierarchicalPWP,
  precision,
  sigmaFromPrecision,
  evaluatePolicy,
} from '../lib/bayesian';
import { GaussianPlot } from './GaussianPlot';
import {
  RotateCcw,
  Bot,
  Sliders,
  Activity,
} from 'lucide-react';

interface ClipSummary {
  utterance_id: string;
  dialog_id: string;
  turn_index: number;
  speaker: string;
  emotion: string;
}

interface CuratedClip {
  utterance_id: string;
  label: string;
  tab_title: string;
  category: string;
  description: string;
  focus_dimension: 'valence' | 'arousal';
}

interface Props {
  apiBase?: string;
}

export const Mode1Explorer: React.FC<Props> = ({ apiBase = '' }) => {
  const [curatedClips, setCuratedClips] = useState<CuratedClip[]>([]);
  const [allClips, setAllClips] = useState<ClipSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>('Ses04F_impro07_M058');
  const [clipData, setClipData] = useState<any>(null);

  const [dimension, setDimension] = useState<'Valence' | 'Arousal'>('Valence');
  const [priorType, setPriorType] = useState<string>('dialogue_context_prior');
  const [modalityMode, setModalityMode] = useState<'multimodal' | 'audio_only' | 'visual_only'>('multimodal');

  // Precisions
  const [tauPrior, setTauPrior] = useState<number>(10.0);
  const [tauAudio, setTauAudio] = useState<number>(15.0);
  const [tauVisual, setTauVisual] = useState<number>(8.0);

  // Model Defaults
  const [defaultTauPrior, setDefaultTauPrior] = useState<number>(10.0);
  const [defaultTauAudio, setDefaultTauAudio] = useState<number>(15.0);
  const [defaultTauVisual, setDefaultTauVisual] = useState<number>(8.0);

  // Fetch clips list on mount (with auto-retry on initial boot)
  useEffect(() => {
    let timeoutId: any;
    let isMounted = true;

    const loadClips = () => {
      fetch(`${apiBase}/api/clips`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((data) => {
          if (!isMounted) return;
          if (data.curated_clips) setCuratedClips(data.curated_clips);
          if (data.all_clips) setAllClips(data.all_clips);
          if (data.curated_clips && data.curated_clips.length > 0) {
            setSelectedId(data.curated_clips[0].utterance_id);
          }
        })
        .catch(() => {
          if (isMounted) {
            timeoutId = setTimeout(loadClips, 2000);
          }
        });
    };

    loadClips();
    return () => {
      isMounted = false;
      clearTimeout(timeoutId);
    };
  }, [apiBase]);

  // Fetch clip details when selectedId changes
  useEffect(() => {
    if (!selectedId) return;
    fetch(`${apiBase}/api/clip/${selectedId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setClipData(data);
        if (data.curated_info?.focus_dimension) {
          setDimension(
            data.curated_info.focus_dimension === 'arousal' ? 'Arousal' : 'Valence'
          );
        }
      })
      .catch((err) => {
        console.error('Failed to load clip details:', err);
      });
  }, [selectedId, apiBase]);

  const dimKey = dimension.toLowerCase();

  // Extract base model distributions from record
  const modelDistributions = useMemo(() => {
    if (!clipData?.record?.distributions) {
      return {
        priorMu: 0.0,
        priorSigma: 0.45,
        audioMu: 0.0,
        audioSigma: 0.35,
        visualMu: 0.0,
        visualSigma: 0.38,
        humanConsensus: clipData?.record?.[`human_${dimKey}`] ?? 0.0,
      };
    }

    const dists = clipData.record.distributions;
    let pMu = 0.0;
    let pSig = 0.45;

    if (priorType === 'dialogue_context_prior' && dists.dialogue_context_prior) {
      pMu = dists.dialogue_context_prior[dimKey]?.mean ?? 0.0;
      pSig = dists.dialogue_context_prior[dimKey]?.sigma ?? 0.45;
    } else if (
      priorType === 'speaker_history_prior_deployable' &&
      dists.speaker_history_prior_deployable
    ) {
      pMu = dists.speaker_history_prior_deployable[dimKey]?.mean ?? 0.0;
      pSig = dists.speaker_history_prior_deployable[dimKey]?.sigma ?? 0.35;
    }

    const aMu = dists.acoustic_likelihood?.[dimKey]?.mean ?? 0.0;
    const aSig = dists.acoustic_likelihood?.[dimKey]?.sigma ?? 0.30;

    const vMu = dists.visual_likelihood?.[dimKey]?.mean ?? 0.0;
    const vSig = dists.visual_likelihood?.[dimKey]?.sigma ?? 0.35;

    const human = clipData.record[`human_${dimKey}`] ?? 0.0;

    return {
      priorMu: pMu,
      priorSigma: pSig,
      audioMu: aMu,
      audioSigma: aSig,
      visualMu: vMu,
      visualSigma: vSig,
      humanConsensus: human,
    };
  }, [clipData, priorType, dimKey]);

  // Sync sliders with model-predicted defaults when clip/dimension/prior changes
  useEffect(() => {
    const tp = Math.min(Math.max(precision(modelDistributions.priorSigma), 0.5), 150);
    const ta = Math.min(Math.max(precision(modelDistributions.audioSigma), 0.5), 150);
    const tv = Math.min(Math.max(precision(modelDistributions.visualSigma), 0.5), 150);

    setDefaultTauPrior(tp);
    setDefaultTauAudio(ta);
    setDefaultTauVisual(tv);

    setTauPrior(tp);
    setTauAudio(ta);
    setTauVisual(tv);
  }, [modelDistributions]);

  // Calculate 60 FPS Exact Posterior using user precisions
  const pwp = useMemo(() => {
    const tA = modalityMode === 'visual_only' ? 0.0001 : tauAudio;
    const tV = modalityMode === 'audio_only' ? 0.0001 : tauVisual;

    return calculateHierarchicalPWP(
      modelDistributions.priorMu,
      tauPrior,
      modelDistributions.audioMu,
      tA,
      modelDistributions.visualMu,
      tV
    );
  }, [modelDistributions, tauPrior, tauAudio, tauVisual, modalityMode]);

  // Evaluate Policy
  const policy = useMemo(() => {
    return evaluatePolicy(
      pwp.priorSensoryStandardizedDiscrepancy,
      pwp.sigmaPost,
      clipData?.record?.turn_index ?? 5
    );
  }, [pwp, clipData]);

  const resetToDefaults = () => {
    setTauPrior(defaultTauPrior);
    setTauAudio(defaultTauAudio);
    setTauVisual(defaultTauVisual);
  };

  const priorDisplayName =
    priorType === 'dialogue_context_prior'
      ? 'Dialogue Context (GRU)'
      : 'Deployable Speaker History';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Scenario Selector Bar */}
      <div className="glass-panel" style={{ padding: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--accent-primary)', fontWeight: 700 }}>
              Mode 1: Dataset Exploration
            </span>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, marginTop: '0.1rem' }}>
              Hierarchical Bayesian Affect Perception
            </h2>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            {/* Prior Type Selector */}
            <select
              value={priorType}
              onChange={(e) => setPriorType(e.target.value)}
              style={{
                padding: '0.4rem 0.75rem',
                borderRadius: '0.4rem',
                fontSize: '0.8rem',
                fontWeight: 600,
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid var(--border-subtle)',
                color: '#ffffff',
              }}
            >
              <option value="dialogue_context_prior">Prior: Dialogue Context (GRU)</option>
              <option value="speaker_history_prior_deployable">Prior: Speaker History (Rolling)</option>
            </select>

            {/* Dimension Selector */}
            <div style={{ display: 'flex', background: 'rgba(255,255,255,0.06)', borderRadius: '0.5rem', padding: '0.25rem' }}>
              {(['Valence', 'Arousal'] as const).map((dim) => (
                <button
                  key={dim}
                  onClick={() => setDimension(dim)}
                  style={{
                    padding: '0.4rem 1rem',
                    borderRadius: '0.35rem',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    border: 'none',
                    cursor: 'pointer',
                    background: dimension === dim ? 'var(--accent-primary)' : 'transparent',
                    color: dimension === dim ? '#ffffff' : '#9ca3af',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {dim}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Curated Regime Chips */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem' }}>
          {curatedClips.map((clip) => {
            const isSelected = clip.utterance_id === selectedId;
            return (
              <button
                key={clip.utterance_id}
                onClick={() => setSelectedId(clip.utterance_id)}
                style={{
                  textAlign: 'left',
                  padding: '0.85rem',
                  borderRadius: '0.65rem',
                  border: isSelected
                    ? '1.5px solid var(--accent-primary)'
                    : '1px solid var(--border-subtle)',
                  background: isSelected
                    ? 'rgba(99, 102, 241, 0.14)'
                    : 'rgba(255, 255, 255, 0.02)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.9rem', color: isSelected ? '#ffffff' : '#e5e7eb' }}>
                    {clip.label}
                  </span>
                  <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.4rem', borderRadius: '4px', background: 'rgba(255,255,255,0.08)', color: '#9ca3af' }}>
                    {clip.focus_dimension}
                  </span>
                </div>
                <p style={{ fontSize: '0.75rem', color: '#9ca3af', lineHeight: 1.35 }}>
                  {clip.description.split('.')[0]}
                </p>
              </button>
            );
          })}
        </div>

        {/* All Clips Search dropdown */}
        <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.8rem', color: '#9ca3af', whiteSpace: 'nowrap' }}>
            Or Search 4,704 IEMOCAP Turns:
          </span>
          <select
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
            style={{
              flex: 1,
              padding: '0.45rem 0.75rem',
              background: 'rgba(0,0,0,0.4)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '0.4rem',
              color: '#ffffff',
              fontSize: '0.85rem',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {allClips.map((c) => (
              <option key={c.utterance_id} value={c.utterance_id}>
                {c.utterance_id} | Turn {c.turn_index} ({c.speaker}) [{c.emotion.toUpperCase()}]
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Left Visualizer & Math / Right Dialogue Context */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.8fr) minmax(0, 1.2fr)', gap: '1.5rem' }}>
        {/* Left Column: Visualizer, Sliders, Math Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Main Visualizer Panel */}
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={18} color="var(--accent-primary)" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                  Gaussian Update Visualizer ({dimension})
                </h3>
              </div>

              {/* Modality Mode Toggle */}
              <div style={{ display: 'flex', background: 'rgba(0,0,0,0.35)', borderRadius: '0.4rem', padding: '0.2rem', gap: '0.2rem' }}>
                <button
                  onClick={() => setModalityMode('multimodal')}
                  style={{
                    padding: '0.25rem 0.65rem',
                    borderRadius: '0.3rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    border: 'none',
                    cursor: 'pointer',
                    background: modalityMode === 'multimodal' ? 'var(--color-sensory)' : 'transparent',
                    color: '#ffffff',
                  }}
                >
                  Multimodal (AV)
                </button>
                <button
                  onClick={() => setModalityMode('audio_only')}
                  style={{
                    padding: '0.25rem 0.65rem',
                    borderRadius: '0.3rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    border: 'none',
                    cursor: 'pointer',
                    background: modalityMode === 'audio_only' ? 'var(--color-audio)' : 'transparent',
                    color: '#ffffff',
                  }}
                >
                  Audio Only (A)
                </button>
                <button
                  onClick={() => setModalityMode('visual_only')}
                  style={{
                    padding: '0.25rem 0.65rem',
                    borderRadius: '0.3rem',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    border: 'none',
                    cursor: 'pointer',
                    background: modalityMode === 'visual_only' ? 'var(--color-visual)' : 'transparent',
                    color: '#ffffff',
                  }}
                >
                  Visual Only (V)
                </button>
              </div>
            </div>

            {/* SVG Gaussian Curve Plot */}
            <GaussianPlot
              muPrior={modelDistributions.priorMu}
              sigmaPrior={sigmaFromPrecision(tauPrior)}
              priorLabel={priorDisplayName}
              muAudio={modelDistributions.audioMu}
              sigmaAudio={sigmaFromPrecision(tauAudio)}
              muVisual={modelDistributions.visualMu}
              sigmaVisual={sigmaFromPrecision(tauVisual)}
              muSensory={pwp.muSensory}
              sigmaSensory={pwp.sigmaSensory}
              muPost={pwp.muPost}
              sigmaPost={pwp.sigmaPost}
              humanConsensus={modelDistributions.humanConsensus}
              dimensionLabel={dimension}
              modalityMode={modalityMode}
            />

            {/* Precision Slider Deck */}
            <div style={{ marginTop: '1.25rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Sliders size={15} color="var(--accent-primary)" />
                  Precision Sliders (Trust Parameters τ = 1/σ²)
                </span>
                <button
                  onClick={resetToDefaults}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    fontSize: '0.75rem',
                    background: 'rgba(255,255,255,0.06)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '0.35rem',
                    color: '#9ca3af',
                    padding: '0.25rem 0.6rem',
                    cursor: 'pointer',
                  }}
                >
                  <RotateCcw size={12} />
                  Reset to Defaults
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '1rem' }}>
                {/* Prior Precision Slider */}
                <div className="glass-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                    <span style={{ color: 'var(--color-prior)', fontWeight: 600 }}>Prior Precision (τ_p)</span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{tauPrior.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="100.0"
                    step="0.5"
                    value={tauPrior}
                    onChange={(e) => setTauPrior(parseFloat(e.target.value))}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                    <span>σ_p = {sigmaFromPrecision(tauPrior).toFixed(3)}</span>
                    <span>Default: {defaultTauPrior.toFixed(1)}</span>
                  </div>
                </div>

                {/* Acoustic Precision Slider */}
                {modalityMode !== 'visual_only' && (
                  <div className="glass-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                      <span style={{ color: 'var(--color-audio)', fontWeight: 600 }}>Acoustic Precision (τ_ℓA)</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{tauAudio.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="100.0"
                      step="0.5"
                      value={tauAudio}
                      onChange={(e) => setTauAudio(parseFloat(e.target.value))}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                      <span>σ_ℓA = {sigmaFromPrecision(tauAudio).toFixed(3)}</span>
                      <span>Default: {defaultTauAudio.toFixed(1)}</span>
                    </div>
                  </div>
                )}

                {/* Visual Precision Slider */}
                {modalityMode !== 'audio_only' && (
                  <div className="glass-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                      <span style={{ color: 'var(--color-visual)', fontWeight: 600 }}>Visual Precision (τ_ℓV)</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{tauVisual.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="100.0"
                      step="0.5"
                      value={tauVisual}
                      onChange={(e) => setTauVisual(parseFloat(e.target.value))}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                      <span>σ_ℓV = {sigmaFromPrecision(tauVisual).toFixed(3)}</span>
                      <span>Default: {defaultTauVisual.toFixed(1)}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 4-Card Layman Metrics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
            {/* Metric 1: Posterior Mean */}
            <div className="glass-panel" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 600 }}>
                Posterior μ_post
              </span>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--color-posterior)', margin: '0.2rem 0' }}>
                {pwp.muPost >= 0 ? `+${pwp.muPost.toFixed(3)}` : pwp.muPost.toFixed(3)}
              </div>
              <p style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                σ_post = {pwp.sigmaPost.toFixed(3)} (τ = {pwp.tauPost.toFixed(1)})
              </p>
            </div>

            {/* Metric 2: 3-Way Source Weights */}
            <div className="glass-panel" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 600 }}>
                3-Way Source Weights
              </span>
              <div style={{ display: 'flex', gap: '0.3rem', alignItems: 'baseline', margin: '0.2rem 0' }}>
                <span style={{ color: 'var(--color-prior)', fontWeight: 700, fontSize: '1.1rem' }}>
                  {(pwp.wPrior * 100).toFixed(0)}%
                </span>
                <span style={{ color: '#6b7280' }}>/</span>
                <span style={{ color: 'var(--color-audio)', fontWeight: 700, fontSize: '1.1rem' }}>
                  {(pwp.wAudio * 100).toFixed(0)}%
                </span>
                <span style={{ color: '#6b7280' }}>/</span>
                <span style={{ color: 'var(--color-visual)', fontWeight: 700, fontSize: '1.1rem' }}>
                  {(pwp.wVisual * 100).toFixed(0)}%
                </span>
              </div>
              {/* Segmented Bar */}
              <div style={{ display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden', marginTop: '0.4rem' }}>
                <div style={{ width: `${pwp.wPrior * 100}%`, background: 'var(--color-prior)' }} />
                <div style={{ width: `${pwp.wAudio * 100}%`, background: 'var(--color-audio)' }} />
                <div style={{ width: `${pwp.wVisual * 100}%`, background: 'var(--color-visual)' }} />
              </div>
              <p style={{ fontSize: '0.68rem', color: '#9ca3af', marginTop: '0.3rem' }}>
                w_C (Context) : w_ℓA (Audio) : w_ℓV (Vision)
              </p>
            </div>

            {/* Metric 3: Standardized Discrepancy */}
            <div className="glass-panel" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 600 }}>
                Standardized Friction (δ*)
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '0.2rem 0' }}>
                <span style={{ fontSize: '1.6rem', fontWeight: 800 }}>
                  {pwp.priorSensoryStandardizedDiscrepancy.toFixed(2)}
                </span>
                <span
                  className={
                    policy.action === 'clarify'
                      ? 'badge-glow-rose'
                      : policy.action === 'cautious'
                      ? 'badge-glow-amber'
                      : 'badge-glow-emerald'
                  }
                  style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', borderRadius: '9999px', fontWeight: 700 }}
                >
                  {pwp.priorSensoryStandardizedDiscrepancy >= 2.0 ? 'High' : pwp.priorSensoryStandardizedDiscrepancy >= 1.0 ? 'Moderate' : 'Low'}
                </span>
              </div>
              <p style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                Cross-Sensory δ*(ℓ_A, ℓ_V) = {pwp.crossSensoryStandardizedDiscrepancy.toFixed(2)}
              </p>
            </div>

            {/* Metric 4: Human Consensus Shift */}
            <div className="glass-panel" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontWeight: 600 }}>
                Human Consensus
              </span>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--color-human)', margin: '0.2rem 0' }}>
                {modelDistributions.humanConsensus >= 0
                  ? `+${modelDistributions.humanConsensus.toFixed(3)}`
                  : modelDistributions.humanConsensus.toFixed(3)}
              </div>
              <p style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                Shift: {(pwp.muPost - modelDistributions.humanConsensus >= 0 ? '+' : '') +
                  (pwp.muPost - modelDistributions.humanConsensus).toFixed(3)}
              </p>
            </div>
          </div>

          {/* Robot Reasoning Policy Card */}
          <div className="glass-panel" style={{ padding: '1.25rem', borderLeft: `4px solid ${policy.action === 'clarify' ? 'var(--accent-rose)' : policy.action === 'cautious' ? 'var(--accent-amber)' : 'var(--accent-emerald)'}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Bot size={18} color="var(--accent-primary)" />
                <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>Robot Reasoning & Policy Gating</span>
              </div>
              <span
                className={
                  policy.action === 'clarify'
                    ? 'badge-glow-rose'
                    : policy.action === 'cautious'
                    ? 'badge-glow-amber'
                    : 'badge-glow-emerald'
                }
                style={{ padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.8rem', fontWeight: 700 }}
              >
                {policy.label}
              </span>
            </div>

            <p style={{ fontSize: '0.85rem', color: '#e5e7eb', marginBottom: '0.5rem' }}>
              {policy.guidance}
            </p>

            <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: '#9ca3af' }}>
              <span>
                <strong>Dispatched iCub Gesture:</strong> <code style={{ color: 'var(--accent-primary)' }}>{policy.gesture}</code>
              </span>
              <span>
                <strong>Active Trigger:</strong> {policy.triggers.join('; ')}
              </span>
            </div>

            {policy.clarificationPrompt && (
              <div style={{ marginTop: '0.75rem', padding: '0.75rem', borderRadius: '0.5rem', background: 'rgba(244, 63, 94, 0.08)', border: '1px solid rgba(244, 63, 94, 0.25)' }}>
                <span style={{ fontSize: '0.75rem', color: '#fb7185', fontWeight: 600 }}>Clarification Dialogue Cue:</span>
                <p style={{ fontSize: '0.85rem', color: '#ffffff', fontStyle: 'italic', marginTop: '0.2rem' }}>
                  "{policy.clarificationPrompt}"
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Dialogue Context, Audio, Transcripts, Raters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Audio & Turn Header Card */}
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-primary)' }}>
                {clipData?.record?.utterance_id}
              </span>
              <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: 'rgba(255,255,255,0.08)' }}>
                {clipData?.record?.emotion?.toUpperCase()}
              </span>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.8rem', color: '#9ca3af', marginBottom: '0.75rem' }}>
              <span>Dialogue: <code>{clipData?.record?.dialog_id}</code></span>
              <span>Turn: <code>{clipData?.record?.turn_index}</code></span>
              <span>Speaker: <code>{clipData?.record?.speaker}</code></span>
            </div>

            {/* Native Audio Player */}
            {clipData?.audio_url && (
              <div style={{ marginTop: '0.5rem' }}>
                <audio
                  controls
                  src={`${apiBase}${clipData.audio_url}`}
                  style={{ width: '100%', height: '36px', borderRadius: '6px' }}
                />
              </div>
            )}
          </div>

          {/* Transcript History */}
          <div className="glass-panel" style={{ padding: '1.25rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.75rem' }}>
              Conversational Context History
            </span>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '340px', overflowY: 'auto', paddingRight: '0.3rem' }}>
              {clipData?.record?.context_history?.map((ctx: any, idx: number) => (
                <div
                  key={idx}
                  style={{
                    padding: '0.5rem 0.75rem',
                    borderRadius: '0.4rem',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderLeft: `2px solid ${ctx.speaker === 'F' ? '#ec4899' : '#3b82f6'}`,
                    fontSize: '0.8rem',
                  }}
                >
                  <div style={{ fontSize: '0.7rem', color: '#9ca3af', marginBottom: '0.15rem' }}>
                    Turn {ctx.turn} — Speaker {ctx.speaker}
                  </div>
                  <div style={{ color: '#e5e7eb' }}>{ctx.text}</div>
                </div>
              ))}

              {/* Target Utterance */}
              <div
                style={{
                  padding: '0.6rem 0.75rem',
                  borderRadius: '0.4rem',
                  background: 'rgba(99, 102, 241, 0.12)',
                  borderLeft: '3px solid var(--accent-primary)',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                }}
              >
                <div style={{ fontSize: '0.7rem', color: 'var(--accent-primary)', marginBottom: '0.15rem' }}>
                  Target Turn {clipData?.record?.turn_index} — Speaker {clipData?.record?.speaker}
                </div>
                <div style={{ color: '#ffffff' }}>"{clipData?.record?.transcript}"</div>
              </div>
            </div>
          </div>

          {/* Per-Rater Annotation Table */}
          {clipData?.record?.per_rater_ratings && (
            <div className="glass-panel" style={{ padding: '1rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, marginBottom: '0.5rem', display: 'block' }}>
                Per-Rater Ground-Truth Annotations
              </span>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(90px, 1fr))', gap: '0.5rem' }}>
                {Object.entries(clipData.record.per_rater_ratings).map(([rid, r]: any) => (
                  <div key={rid} className="glass-card" style={{ padding: '0.4rem 0.6rem', textAlign: 'center' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-human)' }}>{rid}</div>
                    <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>V: {r.valence >= 0 ? `+${r.valence.toFixed(2)}` : r.valence.toFixed(2)}</div>
                    <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>A: {r.arousal >= 0 ? `+${r.arousal.toFixed(2)}` : r.arousal.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
