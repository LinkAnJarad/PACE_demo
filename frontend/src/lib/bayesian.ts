/**
 * Client-Side Exact Bayesian Calculations for 60 FPS Responsive Morphing.
 */

export interface SensoryFusionResult {
  muSensory: number;
  sigmaSensory: number;
  tauAudio: number;
  tauVisual: number;
  tauSensory: number;
  wAudioSensory: number;
  wVisualSensory: number;
  crossSensoryDiscrepancy: number;
  crossSensoryStandardizedDiscrepancy: number;
}

export interface HierarchicalPWPResult {
  muPost: number;
  sigmaPost: number;
  tauPrior: number;
  tauAudio: number;
  tauVisual: number;
  tauSensory: number;
  tauPost: number;
  wPrior: number;
  wAudio: number;
  wVisual: number;
  wSensory: number;
  wAudioSensory: number;
  wVisualSensory: number;
  muSensory: number;
  sigmaSensory: number;
  priorSensoryDiscrepancy: number;
  priorSensoryStandardizedDiscrepancy: number;
  crossSensoryDiscrepancy: number;
  crossSensoryStandardizedDiscrepancy: number;
}

export interface PolicyEvaluation {
  action: 'respond' | 'cautious' | 'clarify';
  label: string;
  badgeColor: 'emerald' | 'amber' | 'rose';
  icon: string;
  gesture: string;
  guidance: string;
  clarificationPrompt?: string;
  triggers: string[];
}

export function precision(sigma: number): number {
  if (sigma <= 0) return 100.0;
  return 1.0 / (sigma * sigma);
}

export function sigmaFromPrecision(tau: number): number {
  if (tau <= 0) return 1.0;
  return 1.0 / Math.sqrt(tau);
}

export function gaussianPdf(x: number, mu: number, sigma: number): number {
  if (sigma <= 0) return 0;
  const coeff = 1.0 / (sigma * Math.sqrt(2.0 * Math.PI));
  const exponent = -0.5 * Math.pow((x - mu) / sigma, 2);
  return coeff * Math.exp(exponent);
}

export function calculateLevel1Sensory(
  muAudio: number,
  tauAudio: number,
  muVisual: number,
  tauVisual: number
): SensoryFusionResult {
  const tauSensory = tauAudio + tauVisual;
  const muSensory = (tauAudio * muAudio + tauVisual * muVisual) / tauSensory;
  const sigmaSensory = sigmaFromPrecision(tauSensory);

  const sigmaAudio = sigmaFromPrecision(tauAudio);
  const sigmaVisual = sigmaFromPrecision(tauVisual);

  const wAudioSensory = tauAudio / tauSensory;
  const wVisualSensory = tauVisual / tauSensory;

  const rawDisc = Math.abs(muAudio - muVisual);
  const jointStd = Math.sqrt(sigmaAudio * sigmaAudio + sigmaVisual * sigmaVisual);
  const standDisc = jointStd > 0 ? rawDisc / jointStd : 0;

  return {
    muSensory,
    sigmaSensory,
    tauAudio,
    tauVisual,
    tauSensory,
    wAudioSensory,
    wVisualSensory,
    crossSensoryDiscrepancy: rawDisc,
    crossSensoryStandardizedDiscrepancy: standDisc,
  };
}

export function calculateHierarchicalPWP(
  muPrior: number,
  tauPrior: number,
  muAudio: number,
  tauAudio: number,
  muVisual: number,
  tauVisual: number
): HierarchicalPWPResult {
  const sensory = calculateLevel1Sensory(muAudio, tauAudio, muVisual, tauVisual);
  const tauPost = tauPrior + sensory.tauSensory;
  const muPost = (tauPrior * muPrior + tauAudio * muAudio + tauVisual * muVisual) / tauPost;
  const sigmaPost = sigmaFromPrecision(tauPost);

  const sigmaPrior = sigmaFromPrecision(tauPrior);

  const wPrior = tauPrior / tauPost;
  const wAudio = tauAudio / tauPost;
  const wVisual = tauVisual / tauPost;
  const wSensory = sensory.tauSensory / tauPost;

  const rawDiscPriorSensory = Math.abs(sensory.muSensory - muPrior);
  const jointStdPriorSensory = Math.sqrt(
    sigmaPrior * sigmaPrior + sensory.sigmaSensory * sensory.sigmaSensory
  );
  const standDiscPriorSensory =
    jointStdPriorSensory > 0 ? rawDiscPriorSensory / jointStdPriorSensory : 0;

  return {
    muPost,
    sigmaPost,
    tauPrior,
    tauAudio,
    tauVisual,
    tauSensory: sensory.tauSensory,
    tauPost,
    wPrior,
    wAudio,
    wVisual,
    wSensory,
    wAudioSensory: sensory.wAudioSensory,
    wVisualSensory: sensory.wVisualSensory,
    muSensory: sensory.muSensory,
    sigmaSensory: sensory.sigmaSensory,
    priorSensoryDiscrepancy: rawDiscPriorSensory,
    priorSensoryStandardizedDiscrepancy: standDiscPriorSensory,
    crossSensoryDiscrepancy: sensory.crossSensoryDiscrepancy,
    crossSensoryStandardizedDiscrepancy: sensory.crossSensoryStandardizedDiscrepancy,
  };
}

export function evaluatePolicy(
  standDisc: number,
  postSigma: number,
  contextTurnCount: number = 5
): PolicyEvaluation {
  const isWarmup = contextTurnCount <= 3;
  const triggers: string[] = [];

  if (standDisc >= 1.5) {
    triggers.push(`High discrepancy (δ* = ${standDisc.toFixed(2)} ≥ 1.5)`);
  } else if (standDisc >= 1.0) {
    triggers.push(`Moderate discrepancy (δ* = ${standDisc.toFixed(2)} ≥ 1.0)`);
  }

  if (postSigma >= 0.36) {
    triggers.push(`High uncertainty (σ_post = ${postSigma.toFixed(2)} ≥ 0.36)`);
  } else if (postSigma >= 0.32) {
    triggers.push(`Moderate uncertainty (σ_post = ${postSigma.toFixed(2)} ≥ 0.32)`);
  }

  if (!isWarmup && (standDisc >= 1.5 || postSigma >= 0.36)) {
    return {
      action: 'clarify',
      label: 'Ask for clarification',
      badgeColor: 'rose',
      icon: 'HelpCircle',
      gesture: 'think',
      guidance: 'Acknowledge ambiguity and invite the speaker to share more context.',
      clarificationPrompt:
        'I may be reading that incorrectly. Would you like to tell me a little more about how that felt?',
      triggers,
    };
  }

  if (!isWarmup && (standDisc >= 1.0 || postSigma >= 0.32)) {
    return {
      action: 'cautious',
      label: 'Respond cautiously',
      badgeColor: 'amber',
      icon: 'Eye',
      gesture: 'tilt_head',
      guidance: 'Phrased tentatively with a head tilt; avoid strong affective attributions.',
      triggers,
    };
  }

  return {
    action: 'respond',
    label: 'Respond normally',
    badgeColor: 'emerald',
    icon: 'CheckCircle2',
    gesture: 'nod',
    guidance: 'The sources are sufficiently aligned and precise to respond confidently.',
    triggers: triggers.length > 0 ? triggers : ['No threshold crossed'],
  };
}
