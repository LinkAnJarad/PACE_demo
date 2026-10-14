import React, { useState } from 'react';

export interface CircumplexDimensionData {
  mu_prior: number;
  sigma_prior: number;
  mu_sensory: number;
  sigma_sensory: number;
  mu_post: number;
  sigma_post: number;
  mu_audio?: number;
  sigma_audio?: number;
  mu_visual?: number;
  sigma_visual?: number;
}

interface AffectiveCircumplexProps {
  valence: CircumplexDimensionData;
  arousal: CircumplexDimensionData;
  height?: number;
  priorLabel?: string;
  showPosterior?: boolean;
  setShowPosterior?: (val: boolean) => void;
  showSensory?: boolean;
  setShowSensory?: (val: boolean) => void;
  showPrior?: boolean;
  setShowPrior?: (val: boolean) => void;
}

export const AffectiveCircumplex: React.FC<AffectiveCircumplexProps> = ({
  valence,
  arousal,
  height: _height,
  priorLabel = 'Prior',
  showPosterior: extShowPost,
  setShowPosterior: extSetShowPost,
  showSensory: extShowSensory,
  setShowSensory: extSetShowSensory,
  showPrior: extShowPrior,
  setShowPrior: extSetShowPrior,
}) => {
  // Local toggle states if not controlled externally
  const [localShowPost, setLocalShowPost] = useState<boolean>(true);
  const [localShowSensory, setLocalShowSensory] = useState<boolean>(false);
  const [localShowPrior, setLocalShowPrior] = useState<boolean>(false);

  const showPost = extShowPost ?? localShowPost;
  const setShowPost = extSetShowPost ?? setLocalShowPost;

  const showSensory = extShowSensory ?? localShowSensory;
  const setShowSensory = extSetShowSensory ?? setLocalShowSensory;

  const showPrior = extShowPrior ?? localShowPrior;
  const setShowPrior = extSetShowPrior ?? setLocalShowPrior;

  const [hoveredPoint, setHoveredPoint] = useState<string | null>(null);

  // Plot Coordinate Range: [-1.5, +1.5]
  const minVal = -1.5;
  const maxVal = 1.5;
  const range = maxVal - minVal;

  // Square 1:1 Aspect Ratio Coordinate System
  const svgWidth = 380;
  const svgHeight = 380;
  const padding = { top: 25, right: 32, bottom: 39, left: 32 };
  const plotWidth = svgWidth - padding.left - padding.right; // 316
  const plotHeight = svgHeight - padding.top - padding.bottom; // 316

  // Map coordinate helpers
  const mapX = (v: number) => {
    const clamped = Math.max(minVal, Math.min(maxVal, v));
    return padding.left + ((clamped - minVal) / range) * plotWidth;
  };

  const mapY = (a: number) => {
    const clamped = Math.max(minVal, Math.min(maxVal, a));
    // SVG Y is inverted (top is positive arousal)
    return padding.top + ((maxVal - clamped) / range) * plotHeight;
  };

  const scaleRadiusX = (sigma: number) => {
    return Math.max(4, (Math.min(1.5, sigma) / range) * plotWidth);
  };

  const scaleRadiusY = (sigma: number) => {
    return Math.max(4, (Math.min(1.5, sigma) / range) * plotHeight);
  };

  const centerX = mapX(0);
  const centerY = mapY(0);

  // Grid tick lines: -1.0, -0.5, 0.5, 1.0
  const ticks = [-1.0, -0.5, 0.5, 1.0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
      {/* Header & Source Toggles */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem',
          fontSize: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{ fontWeight: 700, color: 'var(--text-main)', fontSize: '0.8rem' }}>
            2D Affective Circumplex
          </span>
          <span style={{ color: 'var(--text-dim)', fontSize: '0.7rem' }}>
            (Valence × Arousal)
          </span>
        </div>

        {/* Toggle Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
          {/* Posterior Button (White) */}
          <button
            type="button"
            onClick={() => setShowPost(!showPost)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '0.35rem',
              border: `1px solid ${showPost ? 'rgba(255, 255, 255, 0.7)' : 'rgba(255, 255, 255, 0.1)'}`,
              background: showPost ? 'rgba(255, 255, 255, 0.16)' : 'rgba(255, 255, 255, 0.04)',
              color: showPost ? '#ffffff' : '#6b7280',
              fontSize: '0.72rem',
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
                background: showPost ? '#ffffff' : '#6b7280',
                boxShadow: showPost ? '0 0 6px rgba(255, 255, 255, 0.8)' : 'none',
              }}
            />
            Posterior PWP
          </button>

          {/* Sensory Likelihood Button (Green AV) */}
          <button
            type="button"
            onClick={() => setShowSensory(!showSensory)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '0.35rem',
              border: `1px solid ${showSensory ? 'rgba(16, 185, 129, 0.6)' : 'rgba(255, 255, 255, 0.1)'}`,
              background: showSensory ? 'rgba(16, 185, 129, 0.18)' : 'rgba(255, 255, 255, 0.04)',
              color: showSensory ? '#34d399' : '#6b7280',
              fontSize: '0.72rem',
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
                background: showSensory ? '#10b981' : '#6b7280',
                boxShadow: showSensory ? '0 0 6px #10b981' : 'none',
              }}
            />
            Likelihood (ℓ_AV)
          </button>

          {/* Prior Button (Red) */}
          <button
            type="button"
            onClick={() => setShowPrior(!showPrior)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.2rem 0.55rem',
              borderRadius: '0.35rem',
              border: `1px solid ${showPrior ? 'rgba(239, 68, 68, 0.6)' : 'rgba(255, 255, 255, 0.1)'}`,
              background: showPrior ? 'rgba(239, 68, 68, 0.18)' : 'rgba(255, 255, 255, 0.04)',
              color: showPrior ? '#fca5a5' : '#6b7280',
              fontSize: '0.72rem',
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
                background: showPrior ? '#ef4444' : '#6b7280',
                boxShadow: showPrior ? '0 0 6px #ef4444' : 'none',
              }}
            />
            {priorLabel}
          </button>
        </div>
      </div>

      {/* SVG Canvas */}
      <div
        style={{
          position: 'relative',
          background: 'rgba(5, 8, 16, 0.65)',
          borderRadius: '0.75rem',
          border: '1px solid var(--border-subtle)',
          padding: '0.25rem',
          overflow: 'hidden',
        }}
      >
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{
            width: '100%',
            maxWidth: '360px',
            height: 'auto',
            aspectRatio: '1 / 1',
            display: 'block',
            margin: '0 auto',
          }}
        >
          <defs>
            {/* Subtle Gradient for Posterior Ellipse (White) */}
            <radialGradient id="postEllipseGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.35" />
              <stop offset="60%" stopColor="#ffffff" stopOpacity="0.12" />
              <stop offset="100%" stopColor="#ffffff" stopOpacity="0.01" />
            </radialGradient>

            {/* Subtle Gradient for Sensory Ellipse (Green AV) */}
            <radialGradient id="sensoryEllipseGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
              <stop offset="60%" stopColor="#10b981" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.02" />
            </radialGradient>

            {/* Subtle Gradient for Prior Ellipse (Red) */}
            <radialGradient id="priorEllipseGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.4" />
              <stop offset="60%" stopColor="#ef4444" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.02" />
            </radialGradient>

            {/* Glow Filter */}
            <filter id="glowPost" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background Grid Lines */}
          {ticks.map((t) => (
            <g key={`grid-x-${t}`}>
              <line
                x1={mapX(t)}
                y1={padding.top}
                x2={mapX(t)}
                y2={svgHeight - padding.bottom}
                stroke="rgba(255, 255, 255, 0.05)"
                strokeDasharray="2 3"
                strokeWidth={1}
              />
              <text
                x={mapX(t)}
                y={svgHeight - padding.bottom + 14}
                fill="#6b7280"
                fontSize={9}
                textAnchor="middle"
                fontFamily="var(--font-mono)"
              >
                {t > 0 ? `+${t}` : t}
              </text>
            </g>
          ))}

          {ticks.map((t) => (
            <g key={`grid-y-${t}`}>
              <line
                x1={padding.left}
                y1={mapY(t)}
                x2={svgWidth - padding.right}
                y2={mapY(t)}
                stroke="rgba(255, 255, 255, 0.05)"
                strokeDasharray="2 3"
                strokeWidth={1}
              />
              <text
                x={padding.left - 6}
                y={mapY(t) + 3}
                fill="#6b7280"
                fontSize={9}
                textAnchor="end"
                fontFamily="var(--font-mono)"
              >
                {t > 0 ? `+${t}` : t}
              </text>
            </g>
          ))}

          {/* Zero Axes */}
          <line
            x1={padding.left}
            y1={centerY}
            x2={svgWidth - padding.right}
            y2={centerY}
            stroke="rgba(255, 255, 255, 0.18)"
            strokeWidth={1.5}
          />
          <line
            x1={centerX}
            y1={padding.top}
            x2={centerX}
            y2={svgHeight - padding.bottom}
            stroke="rgba(255, 255, 255, 0.18)"
            strokeWidth={1.5}
          />

          {/* 1. Prior Dot & Ellipse (Red) */}
          {showPrior && (
            <g
              onMouseEnter={() => setHoveredPoint('prior')}
              onMouseLeave={() => setHoveredPoint(null)}
              style={{ cursor: 'pointer' }}
            >
              {/* Uncertainty Ellipse */}
              <ellipse
                cx={mapX(valence.mu_prior)}
                cy={mapY(arousal.mu_prior)}
                rx={scaleRadiusX(valence.sigma_prior)}
                ry={scaleRadiusY(arousal.sigma_prior)}
                fill="url(#priorEllipseGrad)"
                stroke="#ef4444"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                opacity={0.85}
              />
              {/* Center Dot */}
              <circle
                cx={mapX(valence.mu_prior)}
                cy={mapY(arousal.mu_prior)}
                r={5.5}
                fill="#ef4444"
                stroke="#ffffff"
                strokeWidth={1.5}
              />
            </g>
          )}

          {/* 2. Sensory Likelihood Dot & Ellipse (Green AV) */}
          {showSensory && (
            <g
              onMouseEnter={() => setHoveredPoint('sensory')}
              onMouseLeave={() => setHoveredPoint(null)}
              style={{ cursor: 'pointer' }}
            >
              {/* Uncertainty Ellipse */}
              <ellipse
                cx={mapX(valence.mu_sensory)}
                cy={mapY(arousal.mu_sensory)}
                rx={scaleRadiusX(valence.sigma_sensory)}
                ry={scaleRadiusY(arousal.sigma_sensory)}
                fill="url(#sensoryEllipseGrad)"
                stroke="#10b981"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                opacity={0.85}
              />
              {/* Center Dot */}
              <circle
                cx={mapX(valence.mu_sensory)}
                cy={mapY(arousal.mu_sensory)}
                r={5.5}
                fill="#10b981"
                stroke="#ffffff"
                strokeWidth={1.5}
              />
            </g>
          )}

          {/* 3. Posterior Dot & Ellipse (White) */}
          {showPost && (
            <g
              onMouseEnter={() => setHoveredPoint('post')}
              onMouseLeave={() => setHoveredPoint(null)}
              style={{ cursor: 'pointer' }}
            >
              {/* Uncertainty Ellipse */}
              <ellipse
                cx={mapX(valence.mu_post)}
                cy={mapY(arousal.mu_post)}
                rx={scaleRadiusX(valence.sigma_post)}
                ry={scaleRadiusY(arousal.sigma_post)}
                fill="url(#postEllipseGrad)"
                stroke="#ffffff"
                strokeWidth={2}
                opacity={0.95}
              />
              {/* Center Dot with Glow */}
              <circle
                cx={mapX(valence.mu_post)}
                cy={mapY(arousal.mu_post)}
                r={6.5}
                fill="#ffffff"
                stroke="#111827"
                strokeWidth={2}
                filter="url(#glowPost)"
              />
            </g>
          )}

          {/* Axis Title Labels */}
          <text
            x={centerX}
            y={svgHeight - padding.bottom + 22}
            fill="#9ca3af"
            fontSize={10}
            fontWeight={700}
            textAnchor="middle"
          >
            Valence
          </text>
          <text
            x={padding.left - 18}
            y={centerY}
            fill="#9ca3af"
            fontSize={10}
            fontWeight={700}
            textAnchor="middle"
            transform={`rotate(-90, ${padding.left - 18}, ${centerY})`}
          >
            Arousal
          </text>
        </svg>

        {/* Interactive Point Floating Tooltip */}
        {hoveredPoint && (
          <div
            style={{
              position: 'absolute',
              top: '8px',
              right: '8px',
              padding: '0.4rem 0.65rem',
              borderRadius: '0.4rem',
              background: 'rgba(17, 24, 39, 0.92)',
              border: '1px solid var(--border-subtle)',
              backdropFilter: 'blur(8px)',
              fontSize: '0.72rem',
              color: '#ffffff',
              boxShadow: '0 4px 14px rgba(0,0,0,0.4)',
              pointerEvents: 'none',
              lineHeight: 1.4,
            }}
          >
            {hoveredPoint === 'post' && (
              <>
                <div style={{ fontWeight: 800, color: '#ffffff', marginBottom: '0.15rem' }}>
                  ● Posterior PWP
                </div>
                <div>Valence: {valence.mu_post >= 0 ? `+${valence.mu_post.toFixed(2)}` : valence.mu_post.toFixed(2)} (±{valence.sigma_post.toFixed(2)})</div>
                <div>Arousal: {arousal.mu_post >= 0 ? `+${arousal.mu_post.toFixed(2)}` : arousal.mu_post.toFixed(2)} (±{arousal.sigma_post.toFixed(2)})</div>
              </>
            )}
            {hoveredPoint === 'sensory' && (
              <>
                <div style={{ fontWeight: 800, color: '#34d399', marginBottom: '0.15rem' }}>
                  ● Sensory Likelihood (ℓ_AV)
                </div>
                <div>Valence: {valence.mu_sensory >= 0 ? `+${valence.mu_sensory.toFixed(2)}` : valence.mu_sensory.toFixed(2)} (±{valence.sigma_sensory.toFixed(2)})</div>
                <div>Arousal: {arousal.mu_sensory >= 0 ? `+${arousal.mu_sensory.toFixed(2)}` : arousal.mu_sensory.toFixed(2)} (±{arousal.sigma_sensory.toFixed(2)})</div>
              </>
            )}
            {hoveredPoint === 'prior' && (
              <>
                <div style={{ fontWeight: 800, color: '#fca5a5', marginBottom: '0.15rem' }}>
                  ● {priorLabel}
                </div>
                <div>Valence: {valence.mu_prior >= 0 ? `+${valence.mu_prior.toFixed(2)}` : valence.mu_prior.toFixed(2)} (±{valence.sigma_prior.toFixed(2)})</div>
                <div>Arousal: {arousal.mu_prior >= 0 ? `+${arousal.mu_prior.toFixed(2)}` : arousal.mu_prior.toFixed(2)} (±{arousal.sigma_prior.toFixed(2)})</div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
