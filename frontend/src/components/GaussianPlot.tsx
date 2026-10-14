import React, { useMemo, useState } from 'react';
import { gaussianPdf } from '../lib/bayesian';

interface GaussianPlotProps {
  muPrior: number;
  sigmaPrior: number;
  priorLabel?: string;
  muAudio: number;
  sigmaAudio: number;
  muVisual?: number;
  sigmaVisual?: number;
  muSensory?: number;
  sigmaSensory?: number;
  muPost: number;
  sigmaPost: number;
  humanConsensus?: number | null;
  dimensionLabel?: string;
  modalityMode?: 'multimodal' | 'audio_only' | 'visual_only';
  height?: number;
  showPrior?: boolean;
  showAudio?: boolean;
  showVisual?: boolean;
  showSensory?: boolean;
  showPost?: boolean;
}

export const GaussianPlot: React.FC<GaussianPlotProps> = ({
  muPrior,
  sigmaPrior,
  priorLabel = 'Dialogue Context (GRU)',
  muAudio,
  sigmaAudio,
  muVisual,
  sigmaVisual,
  muSensory,
  sigmaSensory,
  muPost,
  sigmaPost,
  humanConsensus,
  dimensionLabel: _dimensionLabel,
  modalityMode = 'multimodal',
  height = 320,
  showPrior = true,
  showAudio = true,
  showVisual = true,
  showSensory = true,
  showPost = true,
}) => {
  const [hoverX, setHoverX] = useState<number | null>(null);

  const xMin = -1.5;
  const xMax = 1.5;
  const numPoints = 300;
  const width = 800;
  const padding = height <= 220
    ? { top: 14, right: 18, bottom: 28, left: 28 }
    : { top: 25, right: 30, bottom: 45, left: 45 };

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Generate curve coordinates
  const { xValues, curves, maxPdf } = useMemo(() => {
    const xs: number[] = [];
    const step = (xMax - xMin) / (numPoints - 1);
    for (let i = 0; i < numPoints; i++) {
      xs.push(xMin + i * step);
    }

    const priorPts = xs.map((x) => gaussianPdf(x, muPrior, sigmaPrior));
    const audioPts = xs.map((x) => gaussianPdf(x, muAudio, sigmaAudio));
    const visualPts =
      muVisual !== undefined && sigmaVisual !== undefined
        ? xs.map((x) => gaussianPdf(x, muVisual, sigmaVisual))
        : [];
    const sensoryPts =
      muSensory !== undefined && sigmaSensory !== undefined
        ? xs.map((x) => gaussianPdf(x, muSensory, sigmaSensory))
        : [];
    const postPts = xs.map((x) => gaussianPdf(x, muPost, sigmaPost));

    let maxVal = Math.max(
      ...priorPts,
      ...audioPts,
      ...visualPts,
      ...sensoryPts,
      ...postPts,
      0.5
    );
    maxVal = maxVal * 1.15; // margin

    return {
      xValues: xs,
      curves: {
        prior: priorPts,
        audio: audioPts,
        visual: visualPts,
        sensory: sensoryPts,
        post: postPts,
      },
      maxPdf: maxVal,
    };
  }, [
    muPrior,
    sigmaPrior,
    muAudio,
    sigmaAudio,
    muVisual,
    sigmaVisual,
    muSensory,
    sigmaSensory,
    muPost,
    sigmaPost,
  ]);

  const mapX = (x: number) => padding.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
  const mapY = (y: number) => padding.top + plotHeight - (y / maxPdf) * plotHeight;

  const makePath = (pdfPoints: number[]) => {
    return pdfPoints
      .map((y, i) => `${i === 0 ? 'M' : 'L'} ${mapX(xValues[i]).toFixed(1)} ${mapY(y).toFixed(1)}`)
      .join(' ');
  };

  const makeAreaPath = (pdfPoints: number[]) => {
    const line = makePath(pdfPoints);
    const bottomY = mapY(0).toFixed(1);
    const startX = mapX(xValues[0]).toFixed(1);
    const endX = mapX(xValues[xValues.length - 1]).toFixed(1);
    return `${line} L ${endX} ${bottomY} L ${startX} ${bottomY} Z`;
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const ratio = (clientX - padding.left) / plotWidth;
    if (ratio >= 0 && ratio <= 1) {
      setHoverX(xMin + ratio * (xMax - xMin));
    }
  };

  const handleMouseLeave = () => setHoverX(null);

  // Colors per paper scheme: Prior=Red, A=Blue, V=Amber, AV=Green, Posterior=White
  const COLOR_PRIOR = '#ef4444';
  const COLOR_AUDIO = '#3b82f6';
  const COLOR_VISUAL = '#f59e0b';
  const COLOR_SENSORY = '#10b981';
  const COLOR_POST = '#ffffff';
  const COLOR_HUMAN = '#06b6d4';

  const isMultimodal = modalityMode === 'multimodal';

  return (
    <div style={{ width: '100%', position: 'relative' }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <defs>
          <linearGradient id="priorGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={COLOR_PRIOR} stopOpacity="0.22" />
            <stop offset="100%" stopColor={COLOR_PRIOR} stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="audioGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={COLOR_AUDIO} stopOpacity="0.18" />
            <stop offset="100%" stopColor={COLOR_AUDIO} stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="visualGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={COLOR_VISUAL} stopOpacity="0.18" />
            <stop offset="100%" stopColor={COLOR_VISUAL} stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="sensoryGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={COLOR_SENSORY} stopOpacity="0.2" />
            <stop offset="100%" stopColor={COLOR_SENSORY} stopOpacity="0.0" />
          </linearGradient>
          <linearGradient id="postGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={COLOR_POST} stopOpacity="0.3" />
            <stop offset="100%" stopColor={COLOR_POST} stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {[-1.0, -0.5, 0.0, 0.5, 1.0].map((tick) => (
          <g key={tick}>
            <line
              x1={mapX(tick)}
              y1={padding.top}
              x2={mapX(tick)}
              y2={padding.top + plotHeight}
              stroke="rgba(255, 255, 255, 0.07)"
              strokeDasharray="4,4"
            />
            <text
              x={mapX(tick)}
              y={padding.top + plotHeight + 20}
              fill="#9ca3af"
              fontSize="11"
              fontFamily="var(--font-mono)"
              textAnchor="middle"
            >
              {tick > 0 ? `+${tick}` : tick}
            </text>
          </g>
        ))}

        {/* Zero baseline */}
        <line
          x1={padding.left}
          y1={mapY(0)}
          x2={padding.left + plotWidth}
          y2={mapY(0)}
          stroke="rgba(255, 255, 255, 0.2)"
          strokeWidth="1.5"
        />

        {/* Shaded Areas */}
        {showPrior && curves.prior.length > 0 && (
          <path d={makeAreaPath(curves.prior)} fill="url(#priorGrad)" />
        )}
        {showAudio && modalityMode !== 'visual_only' && curves.audio.length > 0 && (
          <path d={makeAreaPath(curves.audio)} fill="url(#audioGrad)" />
        )}
        {showVisual && isMultimodal && curves.visual.length > 0 && (
          <path d={makeAreaPath(curves.visual)} fill="url(#visualGrad)" />
        )}
        {showSensory && isMultimodal && curves.sensory.length > 0 && (
          <path d={makeAreaPath(curves.sensory)} fill="url(#sensoryGrad)" />
        )}
        {showPost && curves.post.length > 0 && (
          <path d={makeAreaPath(curves.post)} fill="url(#postGrad)" />
        )}

        {/* Prior Distribution */}
        {showPrior && curves.prior.length > 0 && (
          <path
            d={makePath(curves.prior)}
            fill="none"
            stroke={COLOR_PRIOR}
            strokeWidth="2.2"
            strokeDasharray="4,3"
          />
        )}

        {/* Acoustic Likelihood */}
        {showAudio && modalityMode !== 'visual_only' && curves.audio.length > 0 && (
          <path
            d={makePath(curves.audio)}
            fill="none"
            stroke={COLOR_AUDIO}
            strokeWidth="2.0"
            strokeDasharray={isMultimodal ? '5,3' : undefined}
          />
        )}

        {/* Visual Likelihood */}
        {showVisual && isMultimodal && curves.visual.length > 0 && (
          <path
            d={makePath(curves.visual)}
            fill="none"
            stroke={COLOR_VISUAL}
            strokeWidth="2.0"
            strokeDasharray="5,3"
          />
        )}

        {/* Multimodal Sensory Fusion (AV) */}
        {showSensory && isMultimodal && curves.sensory.length > 0 && (
          <path
            d={makePath(curves.sensory)}
            fill="none"
            stroke={COLOR_SENSORY}
            strokeWidth="2.6"
          />
        )}

        {/* Precision-Weighted Posterior */}
        {showPost && curves.post.length > 0 && (
          <path
            d={makePath(curves.post)}
            fill="none"
            stroke={COLOR_POST}
            strokeWidth="3.0"
            strokeDasharray="8,4"
          />
        )}

        {/* Human Consensus Marker */}
        {humanConsensus !== null && humanConsensus !== undefined && (
          <g>
            <line
              x1={mapX(humanConsensus)}
              y1={padding.top}
              x2={mapX(humanConsensus)}
              y2={padding.top + plotHeight}
              stroke={COLOR_HUMAN}
              strokeWidth="2"
              strokeDasharray="3,3"
            />
            <polygon
              points={`${mapX(humanConsensus) - 6},${mapY(0)} ${mapX(humanConsensus) + 6},${mapY(0)} ${mapX(humanConsensus)},${mapY(0) - 10}`}
              fill={COLOR_HUMAN}
            />
            <text
              x={mapX(humanConsensus)}
              y={padding.top - 8}
              fill={COLOR_HUMAN}
              fontSize="10"
              fontWeight="600"
              textAnchor="middle"
            >
              Human ({humanConsensus.toFixed(2)})
            </text>
          </g>
        )}

        {/* Hover Crosshair */}
        {hoverX !== null && showPost && (
          <g>
            <line
              x1={mapX(hoverX)}
              y1={padding.top}
              x2={mapX(hoverX)}
              y2={padding.top + plotHeight}
              stroke="rgba(255, 255, 255, 0.4)"
              strokeWidth="1"
              strokeDasharray="2,2"
            />
            <circle cx={mapX(hoverX)} cy={mapY(gaussianPdf(hoverX, muPost, sigmaPost))} r="4" fill={COLOR_POST} />
            <text
              x={mapX(hoverX)}
              y={padding.top + 16}
              fill="#ffffff"
              fontSize="10"
              fontFamily="var(--font-mono)"
              textAnchor="middle"
              className="bg-black px-1"
            >
              x = {hoverX.toFixed(2)}
            </text>
          </g>
        )}

        {/* Axis Label */}
        <text
          x={padding.left + plotWidth / 2}
          y={height - 8}
          fill="#9ca3af"
          fontSize="11"
          fontWeight="500"
          textAnchor="middle"
        >
        </text>
      </svg>

      {/* Legend Deck */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.6rem 1.2rem',
          justifyContent: 'center',
          marginTop: '0.5rem',
          fontSize: '0.8rem',
        }}
      >
        {showPrior && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: 12, height: 3, background: COLOR_PRIOR, borderRadius: 2 }} />
            <span style={{ color: '#fca5a5' }}>
              Prior ({priorLabel}): μ={muPrior >= 0 ? `+${muPrior.toFixed(2)}` : muPrior.toFixed(2)}, σ={sigmaPrior.toFixed(2)}
            </span>
          </div>
        )}

        {showAudio && modalityMode !== 'visual_only' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: 12, height: 3, background: COLOR_AUDIO, borderRadius: 2 }} />
            <span style={{ color: '#93c5fd' }}>
              Acoustic Likelihood (ℓ_A): μ={muAudio >= 0 ? `+${muAudio.toFixed(2)}` : muAudio.toFixed(2)}, σ={sigmaAudio.toFixed(2)}
            </span>
          </div>
        )}

        {showVisual && isMultimodal && muVisual !== undefined && sigmaVisual !== undefined && curves.visual.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: 12, height: 3, background: COLOR_VISUAL, borderRadius: 2 }} />
            <span style={{ color: '#fde68a' }}>
              Visual Likelihood (ℓ_V): μ={muVisual >= 0 ? `+${muVisual.toFixed(2)}` : muVisual.toFixed(2)}, σ={sigmaVisual.toFixed(2)}
            </span>
          </div>
        )}

        {showSensory && isMultimodal && muSensory !== undefined && sigmaSensory !== undefined && curves.sensory.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: 14, height: 3, background: COLOR_SENSORY, borderRadius: 2 }} />
            <span style={{ color: '#6ee7b7', fontWeight: 600 }}>
              Sensory Likelihood (ℓ_AV): μ={muSensory >= 0 ? `+${muSensory.toFixed(2)}` : muSensory.toFixed(2)}, σ={sigmaSensory.toFixed(2)}
            </span>
          </div>
        )}

        {showPost && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: 14, height: 3, background: COLOR_POST, borderTop: '2px dashed #ffffff' }} />
            <span style={{ color: '#ffffff', fontWeight: 700 }}>
              PWP Posterior: μ={muPost >= 0 ? `+${muPost.toFixed(2)}` : muPost.toFixed(2)}, σ={sigmaPost.toFixed(2)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
