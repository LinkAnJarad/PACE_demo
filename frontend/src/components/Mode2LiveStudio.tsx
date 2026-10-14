import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  AlertCircle, AudioWaveform, Bot, BrainCircuit, Check, CheckCircle2, Eye, HelpCircle, Layers, MessageSquare, Mic, Radio, RefreshCw, RotateCcw, Server, Settings, Sliders, SlidersHorizontal, Sparkles, Video, Volume2, X, Zap,
} from 'lucide-react';
import { GaussianPlot } from './GaussianPlot';
import { AffectiveCircumplex } from './AffectiveCircumplex';
import { InteractionLogTable, type LoggedTurn } from './InteractionLogTable';
import { precision, sigmaFromPrecision, calculateHierarchicalPWP } from '../lib/bayesian';
import { VadController, type VadState } from '../lib/vad';

interface Props {
  apiBase?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  audioUrl?: string;
  gesture?: string;
  policy?: {
    action: string;
    label: string;
    badge_color: string;
    guidance: string;
    clarification_prompt?: string;
    triggers: string[];
    context_warmup?: boolean;
    context_turn_count?: number;
  };
  dimension_results?: any;
  prior_type?: string;
  turn_result?: any;
}

/**
 * Reconstruct latest turn Bayesian results structure from a recorded LoggedTurn.
 * Used as fallback on undo or page reload.
 */
const loggedTurnToResults = (log: LoggedTurn) => {
  return {
    user_text: log.user_text,
    assistant_text: log.assistant_text,
    selected_gesture: log.gesture,
    policy: {
      action: log.gating_and_policy.action,
      label: log.gating_and_policy.label,
      badge_color:
        log.gating_and_policy.action === 'clarify'
          ? 'rgba(244, 63, 94, 0.12)'
          : log.gating_and_policy.action === 'cautious'
          ? 'rgba(245, 158, 11, 0.12)'
          : 'rgba(16, 185, 129, 0.12)',
      context_warmup: log.gating_and_policy.context_warmup,
      triggers: log.gating_and_policy.triggers,
      delta_max: log.gating_and_policy.delta_max,
      sigma_max: log.gating_and_policy.sigma_max,
    },
    expectation_prior: {
      valence_mu: log.prior.valence.mu,
      valence_sigma: Math.sqrt(Math.max(0.0001, log.prior.valence.var)),
      arousal_mu: log.prior.arousal.mu,
      arousal_sigma: Math.sqrt(Math.max(0.0001, log.prior.arousal.var)),
    },
    acoustic_likelihood: {
      valence_mu: log.likelihood.valence.mu_audio,
      valence_sigma: Math.sqrt(Math.max(0.0001, log.likelihood.valence.var_audio)),
      arousal_mu: log.likelihood.arousal.mu_audio,
      arousal_sigma: Math.sqrt(Math.max(0.0001, log.likelihood.arousal.var_audio)),
    },
    visual_likelihood: {
      valence_mu: log.likelihood.valence.mu_visual,
      valence_sigma: Math.sqrt(Math.max(0.0001, log.likelihood.valence.var_visual)),
      arousal_mu: log.likelihood.arousal.mu_visual,
      arousal_sigma: Math.sqrt(Math.max(0.0001, log.likelihood.arousal.var_visual)),
    },
    dimension_results: {
      valence: {
        mu_sensory: log.likelihood.valence.mu_sensory,
        sigma_sensory: Math.sqrt(Math.max(0.0001, log.likelihood.valence.var_sensory)),
        mu_audio: log.likelihood.valence.mu_audio,
        mu_visual: log.likelihood.valence.mu_visual,
        mu_post: log.posterior.valence.mu,
        sigma_post: log.posterior.valence.sigma,
      },
      arousal: {
        mu_sensory: log.likelihood.arousal.mu_sensory,
        sigma_sensory: Math.sqrt(Math.max(0.0001, log.likelihood.arousal.var_sensory)),
        mu_audio: log.likelihood.arousal.mu_audio,
        mu_visual: log.likelihood.arousal.mu_visual,
        mu_post: log.posterior.arousal.mu,
        sigma_post: log.posterior.arousal.sigma,
      },
    },
    latency_ms: log.latency_ms,
    turn_log: log,
  };
};

export const Mode2LiveStudio: React.FC<Props> = ({ apiBase = '' }) => {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [processing, setProcessing] = useState<boolean>(false);
  const [processingStep, setProcessingStep] = useState<string>('');

  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [priorType, setPriorType] = useState<string>('dialogue_context_prior');
  const [whisperSize, setWhisperSize] = useState<string>(() => {
    return localStorage.getItem('icub_whisper_size') || 'small';
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [llmApiBase, setLlmApiBase] = useState<string>(() => {
    return localStorage.getItem('icub_llm_api_base') || 'http://127.0.0.1:11434';
  });
  const [llmApiKey, setLlmApiKey] = useState<string>(() => {
    return localStorage.getItem('icub_llm_api_key') || '';
  });
  const [fallbackMode, setFallbackMode] = useState<'error' | 'canned'>(() => {
    return (localStorage.getItem('icub_fallback_mode') as 'error' | 'canned') || 'error';
  });
  const [llmConnStatus, setLlmConnStatus] = useState<{
    checking: boolean;
    success: boolean | null;
    message: string;
  }>({
    checking: false,
    success: null,
    message: '',
  });
  const [language] = useState<string>('en');
  const [ttsVoice, setTtsVoice] = useState<string>('af_heart');

  // Turn-Level Interaction Flow Logging State
  const [enableLogging, setEnableLogging] = useState<boolean>(true);
  const [loggedTurns, setLoggedTurns] = useState<LoggedTurn[]>([]);

  // Inspection Dimension & Curve Visibility (1D Gaussian)
  const [dimension, setDimension] = useState<'Valence' | 'Arousal'>('Valence');
  const [showAV, setShowAV] = useState<boolean>(true);
  const [showA, setShowA] = useState<boolean>(true);
  const [showV, setShowV] = useState<boolean>(true);
  const [showPrior, setShowPrior] = useState<boolean>(true);
  const [showPost, setShowPost] = useState<boolean>(true);

  // 2D Circumplex Toggles
  const [circShowPost, setCircShowPost] = useState<boolean>(true);
  const [circShowSensory, setCircShowSensory] = useState<boolean>(false);
  const [circShowPrior, setCircShowPrior] = useState<boolean>(false);

  // Modal State for Precision Controls & Weights
  const [isPrecisionModalOpen, setIsPrecisionModalOpen] = useState<boolean>(false);

  // Interactive Precision Slider Overrides
  const [customTauPrior, setCustomTauPrior] = useState<number | null>(null);
  const [customTauAudio, setCustomTauAudio] = useState<number | null>(null);
  const [customTauVisual, setCustomTauVisual] = useState<number | null>(null);

  const [latestResults, setLatestResults] = useState<any>(null);
  const [resultsHistory, setResultsHistory] = useState<any[]>([]);
  const [shouldAutoplayAudio, setShouldAutoplayAudio] = useState<boolean>(false);

  // Continuous Voice Activity Detection (VAD) State
  const [recordingMode, setRecordingMode] = useState<'manual' | 'vad'>(() => {
    return (localStorage.getItem('icub_recording_mode') as 'manual' | 'vad') || 'manual';
  });
  const [vadSensitivity, setVadSensitivity] = useState<number>(() => {
    return parseInt(localStorage.getItem('icub_vad_sensitivity') || '3', 10);
  });
  const [silenceHangoverMs, setSilenceHangoverMs] = useState<number>(() => {
    return parseInt(localStorage.getItem('icub_vad_hangover') || '850', 10);
  });
  const [vadState, setVadState] = useState<VadState>('IDLE');
  const [audioRms, setAudioRms] = useState<number>(0);
  const [audioThreshold, setAudioThreshold] = useState<number>(0.005);

  // Model-Agnostic LLM State
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    return localStorage.getItem('icub_selected_model') || '';
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const isCancellingRef = useRef<boolean>(false);
  const vadControllerRef = useRef<VadController | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement>(null);

  const checkLlmConnection = async (targetEndpoint?: string) => {
    const endpoint = (targetEndpoint || llmApiBase).trim();
    setLlmConnStatus({ checking: true, success: null, message: 'Connecting to LLM...' });
    try {
      const res = await fetch(`${apiBase}/api/llm/models?api_base=${encodeURIComponent(endpoint)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.models && Array.isArray(data.models) && data.models.length > 0) {
        setAvailableModels(data.models);
        if (!selectedModel || !data.models.includes(selectedModel)) {
          const preferred = data.models.find((m: string) => m.toLowerCase().includes('qwen')) || data.models[0];
          setSelectedModel(preferred);
          localStorage.setItem('icub_selected_model', preferred);
        }
        setLlmConnStatus({
          checking: false,
          success: true,
          message: `Connected (${data.models.length} model${data.models.length === 1 ? '' : 's'}: ${data.models.join(', ')})`,
        });
      } else {
        setLlmConnStatus({
          checking: false,
          success: true,
          message: `Connected to ${data.provider.toUpperCase()}, but no models were listed.`,
        });
      }
    } catch (err: any) {
      setLlmConnStatus({
        checking: false,
        success: false,
        message: `Connection failed: ${err.message || 'Endpoint unreachable'}`,
      });
    }
  };

  // Start Camera and Mic stream
  const startMedia = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: true,
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setCameraActive(true);
    } catch (err) {
      console.error('Camera/Mic access error:', err);
    }
  };

  // Stop Media
  const stopMedia = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setCameraActive(false);
  };

  useEffect(() => {
    startMedia();
    return () => {
      stopMedia();
    };
  }, []);

  // Load existing interaction logs on mount
  useEffect(() => {
    fetch(`${apiBase}/api/live/logs`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (data?.logs && Array.isArray(data.logs)) {
          setLoggedTurns(data.logs);
        }
      })
      .catch(() => {});

    // Discover installed models (Ollama or OpenAI)
    fetch(`${apiBase}/api/llm/models`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (data?.models && Array.isArray(data.models) && data.models.length > 0) {
          setAvailableModels(data.models);
          const saved = localStorage.getItem('icub_selected_model');
          if (saved && data.models.includes(saved)) {
            setSelectedModel(saved);
          } else {
            const preferred = data.models.find((m: string) => m.toLowerCase().includes('qwen')) || data.models[0];
            setSelectedModel(preferred);
          }
        } else if (data?.active_model) {
          setAvailableModels([data.active_model]);
          setSelectedModel((prev) => prev || data.active_model);
        }
      })
      .catch(() => {});
  }, [apiBase]);

  // One-Click Synchronized MediaRecorder
  const startRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      return;
    }
    if (!stream) {
      startMedia();
      return;
    }

    recordedChunksRef.current = [];
    isCancellingRef.current = false;
    try {
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
          ? 'video/webm;codecs=vp8,opus'
          : 'video/webm',
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        if (isCancellingRef.current) {
          isCancellingRef.current = false;
          return;
        }
        const fullBlob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        if (fullBlob.size === 0) {
          console.warn('Recorded video chunk is empty.');
          return;
        }
        sendTurnToServer(fullBlob);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100); // 100ms chunk timeslice
      setIsRecording(true);
    } catch (err) {
      console.error('MediaRecorder start failed:', err);
      if (recordingMode === 'manual') {
        alert('Could not start video recording.');
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const cancelRecording = () => {
    isCancellingRef.current = true;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
    if (vadControllerRef.current) {
      vadControllerRef.current.setProcessing(false);
      vadControllerRef.current.setRobotSpeaking(false);
    }
  };

  // Continuous VAD Controller Lifecycle
  useEffect(() => {
    if (!stream || !cameraActive || recordingMode !== 'vad') {
      if (vadControllerRef.current) {
        vadControllerRef.current.destroy();
        vadControllerRef.current = null;
        setVadState('IDLE');
        setAudioRms(0);
      }
      return;
    }

    const vad = new VadController(stream, {
      sensitivity: vadSensitivity,
      silenceHangoverMs: silenceHangoverMs,
      onSpeechStart: () => {
        startRecording();
      },
      onSpeechEnd: () => {
        stopRecording();
      },
      onRmsUpdate: (rms, thresh) => {
        setAudioRms(rms);
        setAudioThreshold(thresh);
      },
      onStateChange: (newState) => {
        setVadState(newState);
      },
    });

    vad.start();
    vadControllerRef.current = vad;

    return () => {
      vad.destroy();
      vadControllerRef.current = null;
    };
  }, [stream, cameraActive, recordingMode]);

  useEffect(() => {
    vadControllerRef.current?.setSensitivity(vadSensitivity);
    localStorage.setItem('icub_vad_sensitivity', vadSensitivity.toString());
  }, [vadSensitivity]);

  useEffect(() => {
    vadControllerRef.current?.setSilenceHangoverMs(silenceHangoverMs);
    localStorage.setItem('icub_vad_hangover', silenceHangoverMs.toString());
  }, [silenceHangoverMs]);

  useEffect(() => {
    localStorage.setItem('icub_recording_mode', recordingMode);
  }, [recordingMode]);

  // Submit Turn to FastAPI Backend
  const sendTurnToServer = async (videoBlob: Blob) => {
    if (!videoBlob || videoBlob.size === 0) {
      console.warn('Empty video blob received.');
      return;
    }

    setProcessing(true);
    vadControllerRef.current?.setProcessing(true);
    setProcessingStep('Transcribing speech & extracting multimodal affect...');

    // Sanitize chat history to avoid sending large base64 audio strings in form data
    const sanitizedHistory = chatHistory.map((m) => ({
      role: m.role,
      text: m.text,
      dimension_results: m.dimension_results
        ? {
            valence: {
              mu_sensory: m.dimension_results.valence?.mu_sensory,
              mu_audio: m.dimension_results.valence?.mu_audio,
              mu_visual: m.dimension_results.valence?.mu_visual,
            },
            arousal: {
              mu_sensory: m.dimension_results.arousal?.mu_sensory,
              mu_audio: m.dimension_results.arousal?.mu_audio,
              mu_visual: m.dimension_results.arousal?.mu_visual,
            },
          }
        : undefined,
    }));

    const nextTurnIdx = loggedTurns.length > 0 ? Math.max(...loggedTurns.map((l) => l.turn_idx)) + 1 : 1;

    const formData = new FormData();
    formData.append('video', videoBlob, 'turn.webm');
    formData.append('chat_history', JSON.stringify(sanitizedHistory));
    formData.append('prior_type', priorType);
    formData.append('whisper_size', whisperSize);
    formData.append('language', language);
    formData.append('tts_voice', ttsVoice);
    formData.append('enable_logging', enableLogging.toString());
    formData.append('turn_idx', nextTurnIdx.toString());
    if (selectedModel) {
      formData.append('llm_model', selectedModel);
    }

    try {
      setProcessingStep('Computing Level-1 Fusion & Level-2 PWP Integration...');
      const res = await fetch(`${apiBase}/api/live/process-turn`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      }

      const data = await res.json();
      setLatestResults(data);
      setResultsHistory((prev) => [...prev, data]);
      setShouldAutoplayAudio(true);

      // Add to interaction flow logs if logging is active
      if (data.turn_log && enableLogging) {
        setLoggedTurns((prev) => {
          const exists = prev.some((t) => t.turn_idx === data.turn_log.turn_idx);
          if (exists) {
            return prev.map((t) => (t.turn_idx === data.turn_log.turn_idx ? data.turn_log : t));
          }
          return [...prev, data.turn_log];
        });
      }

      // Add to conversation history
      const userMsg: ChatMessage = {
        role: 'user',
        text: data.user_text,
      };

      const assistantMsg: ChatMessage = {
        role: 'assistant',
        text: data.assistant_text,
        audioUrl: data.assistant_audio_b64,
        gesture: data.selected_gesture,
        policy: data.policy,
        dimension_results: data.dimension_results,
        prior_type: priorType,
        turn_result: data,
      };

      setChatHistory((prev) => [...prev, userMsg, assistantMsg]);
    } catch (err) {
      console.error('Live turn processing failed:', err);
      alert(`Live turn execution failed: ${err}`);
    } finally {
      setProcessing(false);
      vadControllerRef.current?.setProcessing(false);
      setProcessingStep('');
    }
  };

  const clearChat = () => {
    setChatHistory([]);
    setResultsHistory([]);
    setLatestResults(null);
    setCustomTauPrior(null);
    setCustomTauAudio(null);
    setCustomTauVisual(null);
    setShouldAutoplayAudio(false);
    handleClearLogs();
    try {
      fetch(`${apiBase}/reset`, { method: 'POST' }).catch(() => {});
    } catch {
      // Ignore network errors on reset
    }
  };

  const handleClearLogs = async () => {
    try {
      await fetch(`${apiBase}/api/live/clear-logs`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to clear logs on server:', e);
    }
    setLoggedTurns([]);
  };

  const handleDeleteLastTurn = async () => {
    if (loggedTurns.length === 0 && chatHistory.length === 0 && resultsHistory.length === 0) return;

    try {
      await fetch(`${apiBase}/api/live/delete-last-log`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to delete last turn on server:', e);
    }

    // 1. Remove last logged turn
    const remainingLogs = loggedTurns.slice(0, -1);
    setLoggedTurns(remainingLogs);

    // 2. Roll back chat history (remove the last assistant message and its matching user message)
    const remainingChat = [...chatHistory];
    if (remainingChat.length > 0 && remainingChat[remainingChat.length - 1].role === 'assistant') {
      remainingChat.pop();
    }
    if (remainingChat.length > 0 && remainingChat[remainingChat.length - 1].role === 'user') {
      remainingChat.pop();
    }
    setChatHistory(remainingChat);

    // 3. Roll back results history and identify the previous turn's result
    let prevTurnResult: any = null;
    const remainingResults = resultsHistory.slice(0, -1);
    setResultsHistory(remainingResults);

    if (remainingResults.length > 0) {
      prevTurnResult = remainingResults[remainingResults.length - 1];
    } else {
      // Fallback 1: check remainingChat for assistant message with turn_result
      const lastAssistant = [...remainingChat].reverse().find((m) => m.role === 'assistant' && m.turn_result);
      if (lastAssistant && lastAssistant.turn_result) {
        prevTurnResult = lastAssistant.turn_result;
      } else if (remainingLogs.length > 0) {
        // Fallback 2: reconstruct from remainingLogs
        prevTurnResult = loggedTurnToResults(remainingLogs[remainingLogs.length - 1]);
      }
    }

    // Reset custom precision slider overrides so the previous turn uses its natural values
    resetPrecisionSliders();
    setShouldAutoplayAudio(false);

    // 4. Restore latest turn Bayesian results view to previous turn (or null if all turns undone)
    setLatestResults(prevTurnResult);

    // 5. Smoothly reset robot simulator posture
    try {
      fetch(`${apiBase}/reset`, { method: 'POST' }).catch(() => {});
    } catch {
      // Ignore network errors on reset
    }
  };

  const resetPrecisionSliders = () => {
    setCustomTauPrior(null);
    setCustomTauAudio(null);
    setCustomTauVisual(null);
  };

  // Compute Bayesian PWP for both dimensions simultaneously
  const { valenceData, arousalData } = useMemo(() => {
    if (!latestResults) {
      return { valenceData: null, arousalData: null };
    }

    const compute = (dim: 'valence' | 'arousal') => {
      const aMu = latestResults.acoustic_likelihood?.[`${dim}_mu`] ?? 0.0;
      const aSig = latestResults.acoustic_likelihood?.[`${dim}_sigma`] ?? 0.40;
      const vMu = latestResults.visual_likelihood?.[`${dim}_mu`] ?? 0.0;
      const vSig = latestResults.visual_likelihood?.[`${dim}_sigma`] ?? 0.40;

      let pMu = 0.0;
      let pSig = 0.50;
      let pName = 'Dialogue Context C (GRU)';

      if (priorType === 'dialogue_context_prior' || priorType === 'C') {
        pMu = latestResults.expectation_prior?.[`${dim}_mu`] ?? 0.0;
        pSig = latestResults.expectation_prior?.[`${dim}_sigma`] ?? 0.50;
        pName = 'Dialogue Context C (GRU)';
      } else if (priorType === 'speaker_history_prior_deployable_multimodal' || priorType === 'H(AV)' || priorType === 'H(ℓ_AV)') {
        const pastValues: number[] = [];
        chatHistory.forEach((msg) => {
          if (msg.role === 'assistant' && msg.dimension_results?.[dim]) {
            pastValues.push(msg.dimension_results[dim].mu_sensory);
          }
        });
        if (pastValues.length > 0) {
          pMu = pastValues.reduce((a, b) => a + b, 0) / pastValues.length;
          if (pastValues.length > 1) {
            const mean = pMu;
            const variance = pastValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / pastValues.length;
            pSig = Math.max(0.18, Math.sqrt(variance));
          } else {
            pSig = 0.35;
          }
        } else {
          pMu = 0.0;
          pSig = 0.50;
        }
        pName = 'Speaker History H(ℓ_AV)';
      } else if (priorType === 'speaker_history_prior_deployable' || priorType === 'H(A)' || priorType === 'H(ℓ_A)') {
        const pastValues: number[] = [];
        chatHistory.forEach((msg) => {
          if (msg.role === 'assistant' && msg.dimension_results?.[dim]) {
            pastValues.push(msg.dimension_results[dim].mu_audio);
          }
        });
        if (pastValues.length > 0) {
          pMu = pastValues.reduce((a, b) => a + b, 0) / pastValues.length;
          if (pastValues.length > 1) {
            const mean = pMu;
            const variance = pastValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / pastValues.length;
            pSig = Math.max(0.18, Math.sqrt(variance));
          } else {
            pSig = 0.35;
          }
        } else {
          pMu = 0.0;
          pSig = 0.50;
        }
        pName = 'Speaker History H(ℓ_A)';
      } else if (priorType === 'speaker_history_prior_deployable_visual' || priorType === 'H(V)' || priorType === 'H(ℓ_V)') {
        const pastValues: number[] = [];
        chatHistory.forEach((msg) => {
          if (msg.role === 'assistant' && msg.dimension_results?.[dim]) {
            pastValues.push(msg.dimension_results[dim].mu_visual);
          }
        });
        if (pastValues.length > 0) {
          pMu = pastValues.reduce((a, b) => a + b, 0) / pastValues.length;
          if (pastValues.length > 1) {
            const mean = pMu;
            const variance = pastValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / pastValues.length;
            pSig = Math.max(0.18, Math.sqrt(variance));
          } else {
            pSig = 0.35;
          }
        } else {
          pMu = 0.0;
          pSig = 0.50;
        }
        pName = 'Speaker History H(ℓ_V)';
      }

      const baseTauP = precision(pSig);
      const baseTauA = precision(aSig);
      const baseTauV = precision(vSig);

      const activeTauP = customTauPrior ?? baseTauP;
      const activeTauA = customTauAudio ?? baseTauA;
      const activeTauV = customTauVisual ?? baseTauV;

      const activeSigmaP = sigmaFromPrecision(activeTauP);
      const activeSigmaA = sigmaFromPrecision(activeTauA);
      const activeSigmaV = sigmaFromPrecision(activeTauV);

      const pwp = calculateHierarchicalPWP(pMu, activeTauP, aMu, activeTauA, vMu, activeTauV);
      const isCustomized = customTauPrior !== null || customTauAudio !== null || customTauVisual !== null;

      return {
        ...pwp,
        mu_prior: pMu,
        sigma_prior: activeSigmaP,
        prior_name: pName,
        mu_audio: aMu,
        sigma_audio: activeSigmaA,
        mu_visual: vMu,
        sigma_visual: activeSigmaV,
        mu_sensory: pwp.muSensory,
        sigma_sensory: pwp.sigmaSensory,
        mu_post: pwp.muPost,
        sigma_post: pwp.sigmaPost,
        baseTauP,
        baseTauA,
        baseTauV,
        isCustomized,
      };
    };

    return {
      valenceData: compute('valence'),
      arousalData: compute('arousal'),
    };
  }, [
    latestResults,
    priorType,
    chatHistory,
    customTauPrior,
    customTauAudio,
    customTauVisual,
  ]);

  const activeBayesianData = dimension === 'Valence' ? valenceData : arousalData;

  // Handlers for precision sliders
  const handleTauPriorChange = (val: number) => {
    setCustomTauPrior(val);
  };

  const handleTauAudioChange = (val: number) => {
    setCustomTauAudio(val);
    if (customTauVisual === null && activeBayesianData) {
      setCustomTauVisual(activeBayesianData.baseTauV);
    }
  };

  const handleTauVisualChange = (val: number) => {
    setCustomTauVisual(val);
    if (customTauAudio === null && activeBayesianData) {
      setCustomTauAudio(activeBayesianData.baseTauA);
    }
  };

  const handleTauSensoryChange = (newSensoryTau: number) => {
    if (!activeBayesianData) return;
    const currentA = customTauAudio ?? activeBayesianData.baseTauA;
    const currentV = customTauVisual ?? activeBayesianData.baseTauV;
    const currentSum = currentA + currentV;
    const ratioA = currentSum > 0 ? currentA / currentSum : 0.5;
    const newA = Math.max(0.2, newSensoryTau * ratioA);
    const newV = Math.max(0.2, newSensoryTau * (1 - ratioA));
    setCustomTauAudio(parseFloat(newA.toFixed(2)));
    setCustomTauVisual(parseFloat(newV.toFixed(2)));
  };

  // Extract user transcripts for dedicated user feed
  const userTurns = useMemo(() => {
    return chatHistory
      .map((msg, idx) => ({ ...msg, originalIndex: idx }))
      .filter((msg) => msg.role === 'user');
  }, [chatHistory]);

  // Extract latest assistant message for robot output card
  const latestAssistantTurn = useMemo(() => {
    for (let i = chatHistory.length - 1; i >= 0; i--) {
      if (chatHistory[i].role === 'assistant') {
        return chatHistory[i];
      }
    }
    return null;
  }, [chatHistory]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Top Header & Settings Control Panel */}
      <div className="glass-panel" style={{ padding: '0.85rem 1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--accent-primary)', fontWeight: 700 }}>
              Live Multimodal Interaction Studio
            </span>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800, marginTop: '0.05rem', color: '#ffffff' }}>
              Real-Time Human–Robot Predictive Affect Processing
            </h2>
          </div>

          {/* Interactive Controls Deck */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
            {/* Prior Selection */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.78rem', color: '#9ca3af', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                <Layers size={13} color="var(--accent-primary)" />
                Prior:
              </span>
              <select
                value={priorType}
                onChange={(e) => setPriorType(e.target.value)}
                style={{
                  padding: '0.35rem 0.65rem',
                  borderRadius: '0.4rem',
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid var(--border-subtle)',
                  color: '#93c5fd',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  outline: 'none',
                  cursor: 'pointer',
                }}
              >
                <option value="dialogue_context_prior" style={{ background: '#111827' }}>
                  C: Dialogue Context (GRU)
                </option>
                <option value="speaker_history_prior_deployable_multimodal" style={{ background: '#111827' }}>
                  H(ℓ_AV): Multimodal Speaker History
                </option>
                <option value="speaker_history_prior_deployable" style={{ background: '#111827' }}>
                  H(ℓ_A): Acoustic Speaker History
                </option>
                <option value="speaker_history_prior_deployable_visual" style={{ background: '#111827' }}>
                  H(ℓ_V): Visual Speaker History
                </option>
              </select>
            </div>

            {/* English Voice Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ fontSize: '0.78rem', color: '#9ca3af', fontWeight: 600 }}>Voice:</span>
              <select
                value={ttsVoice}
                onChange={(e) => setTtsVoice(e.target.value)}
                style={{
                  padding: '0.35rem 0.65rem',
                  borderRadius: '0.4rem',
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid var(--border-subtle)',
                  color: 'var(--text-main)',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  outline: 'none',
                  cursor: 'pointer',
                }}
              >
                <option value="af_heart" style={{ background: '#111827' }}>af_heart (Female US - Warm)</option>
                <option value="af_bella" style={{ background: '#111827' }}>af_bella (Female US - Energetic)</option>
                <option value="af_sarah" style={{ background: '#111827' }}>af_sarah (Female US - Soft)</option>
                <option value="am_fenrir" style={{ background: '#111827' }}>am_fenrir (Male US - Resonant)</option>
                <option value="am_puck" style={{ background: '#111827' }}>am_puck (Male US - Dynamic)</option>
                <option value="bf_alice" style={{ background: '#111827' }}>bf_alice (Female UK - Gentle)</option>
                <option value="bm_fable" style={{ background: '#111827' }}>bm_fable (Male UK - Conversational)</option>
              </select>
            </div>

            {/* Model-Agnostic LLM Selector */}
            {availableModels.length > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{ fontSize: '0.78rem', color: '#9ca3af', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <Bot size={13} color="#c084fc" />
                  Model:
                </span>
                <select
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    localStorage.setItem('icub_selected_model', e.target.value);
                  }}
                  style={{
                    padding: '0.35rem 0.65rem',
                    borderRadius: '0.4rem',
                    background: 'rgba(255,255,255,0.06)',
                    border: '1px solid rgba(192, 132, 252, 0.4)',
                    color: '#e9d5ff',
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    outline: 'none',
                    cursor: 'pointer',
                  }}
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m} style={{ background: '#111827' }}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Settings Sidebar Toggle Button */}
            <button
              type="button"
              onClick={() => setIsSettingsOpen(true)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '0.35rem 0.75rem',
                borderRadius: '0.4rem',
                background: isSettingsOpen ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255,255,255,0.06)',
                border: isSettingsOpen ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                color: isSettingsOpen ? '#a5b4fc' : '#d1d5db',
                fontSize: '0.78rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <Settings size={13} color="var(--accent-primary)" />
              Settings
            </button>

            {/* Precision & Weights Pop-up Button */}
            <button
              type="button"
              onClick={() => setIsPrecisionModalOpen(true)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '0.35rem 0.75rem',
                borderRadius: '0.4rem',
                background: activeBayesianData?.isCustomized ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255,255,255,0.06)',
                border: activeBayesianData?.isCustomized ? '1px solid var(--accent-primary)' : '1px solid var(--border-subtle)',
                color: activeBayesianData?.isCustomized ? '#a5b4fc' : '#d1d5db',
                fontSize: '0.78rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <SlidersHorizontal size={13} color="var(--accent-primary)" />
              Precision & Weights
              {activeBayesianData?.isCustomized && (
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#6366f1' }} />
              )}
            </button>

            {/* Reset Button */}
            <button
              type="button"
              onClick={clearChat}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                padding: '0.35rem 0.75rem',
                borderRadius: '0.4rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid var(--border-subtle)',
                color: '#9ca3af',
                fontSize: '0.78rem',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              <RotateCcw size={13} />
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* 3-Column Studio Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(310px, 1fr) minmax(420px, 1.4fr) minmax(310px, 1.1fr)',
          gap: '1.25rem',
          alignItems: 'start',
        }}
      >
        {/* ── LEFT COLUMN (User / Input) ─────────────────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* 1. Enlarged Webcam View */}
          <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Video size={15} color="var(--accent-primary)" />
                Webcam View (User Input)
              </span>
              <span
                style={{
                  fontSize: '0.68rem',
                  padding: '0.12rem 0.5rem',
                  borderRadius: '9999px',
                  background: cameraActive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                  color: cameraActive ? '#34d399' : '#fb7185',
                  fontWeight: 600,
                }}
              >
                {cameraActive ? 'Camera Live' : 'Offline'}
              </span>
            </div>

            {/* Video Viewport Container */}
            <div
              style={{
                position: 'relative',
                width: '100%',
                height: '240px',
                borderRadius: '0.6rem',
                overflow: 'hidden',
                background: '#000000',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
              />

              {/* Facial Reticle Guide */}
              <div
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: '58%',
                  height: '75%',
                  border: isRecording ? '2px dashed #ef4444' : '1px dashed rgba(255, 255, 255, 0.3)',
                  borderRadius: '50%',
                  pointerEvents: 'none',
                  boxShadow: isRecording ? '0 0 16px rgba(239, 68, 68, 0.6)' : 'none',
                }}
              />

              {/* Recording Indicator */}
              {isRecording && (
                <div
                  style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    padding: '0.2rem 0.55rem',
                    borderRadius: '4px',
                    background: 'rgba(239, 68, 68, 0.95)',
                    color: '#ffffff',
                    fontSize: '0.68rem',
                    fontWeight: 800,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
                  }}
                >
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#ffffff', animation: 'pulse-slow 1s infinite' }} />
                  RECORDING MULTIMODAL TURN
                </div>
              )}
            </div>
          </div>

          {/* 2. User Dialogue Transcript Stream */}
          <div className="glass-panel" style={{ padding: '1rem', height: '340px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <MessageSquare size={14} color="var(--accent-primary)" />
                User Dialogue Transcript
              </span>
              <span style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                {userTurns.length} spoken turn{userTurns.length === 1 ? '' : 's'}
              </span>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.65rem', paddingRight: '0.3rem' }}>
              {userTurns.length === 0 ? (
                <div style={{ textAlign: 'center', color: '#6b7280', margin: 'auto', fontSize: '0.82rem', padding: '1rem' }}>
                  💬 No user speech recorded yet. Use the record button on the right to start speaking.
                </div>
              ) : (
                userTurns.map((turn, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '0.75rem 0.9rem',
                      borderRadius: '0.6rem',
                      background: 'rgba(99, 102, 241, 0.12)',
                      border: '1px solid rgba(99, 102, 241, 0.3)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                      <span style={{ fontSize: '0.68rem', color: '#a5b4fc', fontWeight: 700 }}>
                        👤 Spoken Turn #{idx + 1}
                      </span>
                    </div>
                    <div style={{ color: '#ffffff', fontSize: '0.88rem', lineHeight: 1.45 }}>
                      &ldquo;{turn.text}&rdquo;
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* ── CENTER COLUMN (Predictive Mind / Bayesian Brain) ─────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {!activeBayesianData || !valenceData || !arousalData ? (
            /* Standby State */
            <div
              className="glass-panel"
              style={{
                padding: '3.5rem 2rem',
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '1.25rem',
                minHeight: '580px',
              }}
            >
              <div
                style={{
                  width: 56,
                  height: 56,
                  borderRadius: '50%',
                  background: 'rgba(99, 102, 241, 0.12)',
                  border: '1px solid rgba(99, 102, 241, 0.3)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 0 20px rgba(99, 102, 241, 0.2)',
                }}
              >
                <BrainCircuit size={28} color="var(--accent-primary)" />
              </div>
              <div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.35rem' }}>
                  Bayesian Brain on Standby
                </h3>
                <p style={{ fontSize: '0.82rem', color: '#9ca3af', maxWidth: '380px', lineHeight: 1.55 }}>
                  Click &ldquo;Speak to Robot&rdquo; on the right to capture speech + visual cues and observe live Bayesian updates (A + V → AV → PWP).
                </p>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
                <span style={{ fontSize: '0.72rem', padding: '0.2rem 0.55rem', borderRadius: '9999px', background: 'rgba(239, 68, 68, 0.12)', color: '#fca5a5', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                  Prior: {priorType === 'dialogue_context_prior' ? 'Dialogue Context C' : 'Speaker History H'}
                </span>
                <span style={{ fontSize: '0.72rem', padding: '0.2rem 0.55rem', borderRadius: '9999px', background: 'rgba(59, 130, 246, 0.12)', color: '#93c5fd', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                  Acoustic (Wav2Vec2)
                </span>
                <span style={{ fontSize: '0.72rem', padding: '0.2rem 0.55rem', borderRadius: '9999px', background: 'rgba(245, 158, 11, 0.12)', color: '#fde68a', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                  Visual (EmoNet)
                </span>
              </div>
            </div>
          ) : (
            <>
              {/* 1. 1D GAUSSIAN DENSITY PLOT (Retained on Top) */}
              <div className="glass-panel" style={{ padding: '0.9rem 1.1rem' }}>
                {/* 1D Plot Toolbar: Dimension Switcher & Curve Checkboxes */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '0.5rem',
                    marginBottom: '0.5rem',
                    borderBottom: '1px solid var(--border-subtle)',
                    paddingBottom: '0.5rem',
                  }}
                >
                  {/* Dimension Switcher */}
                  <div style={{ display: 'flex', background: 'rgba(255,255,255,0.06)', borderRadius: '0.4rem', padding: '0.15rem' }}>
                    <button
                      type="button"
                      onClick={() => setDimension('Valence')}
                      style={{
                        padding: '0.25rem 0.7rem',
                        borderRadius: '0.35rem',
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        border: 'none',
                        cursor: 'pointer',
                        background: dimension === 'Valence' ? 'var(--accent-primary)' : 'transparent',
                        color: dimension === 'Valence' ? '#ffffff' : '#9ca3af',
                      }}
                    >
                      Valence
                    </button>
                    <button
                      type="button"
                      onClick={() => setDimension('Arousal')}
                      style={{
                        padding: '0.25rem 0.7rem',
                        borderRadius: '0.35rem',
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        border: 'none',
                        cursor: 'pointer',
                        background: dimension === 'Arousal' ? 'var(--accent-primary)' : 'transparent',
                        color: dimension === 'Arousal' ? '#ffffff' : '#9ca3af',
                      }}
                    >
                      Arousal
                    </button>
                  </div>

                  {/* Curve Visibility Checkboxes */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap', fontSize: '0.72rem' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#fca5a5', cursor: 'pointer' }}>
                      <input type="checkbox" checked={showPrior} onChange={(e) => setShowPrior(e.target.checked)} />
                      Prior
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#93c5fd', cursor: 'pointer' }}>
                      <input type="checkbox" checked={showA} onChange={(e) => setShowA(e.target.checked)} />
                      Audio
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#fde68a', cursor: 'pointer' }}>
                      <input type="checkbox" checked={showV} onChange={(e) => setShowV(e.target.checked)} />
                      Visual
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#34d399', cursor: 'pointer' }}>
                      <input type="checkbox" checked={showAV} onChange={(e) => setShowAV(e.target.checked)} />
                      Sensory AV
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#ffffff', fontWeight: 700, cursor: 'pointer' }}>
                      <input type="checkbox" checked={showPost} onChange={(e) => setShowPost(e.target.checked)} />
                      Posterior
                    </label>
                  </div>
                </div>

                {/* 1D Gaussian Plot Component */}
                <GaussianPlot
                  muPrior={activeBayesianData.mu_prior}
                  sigmaPrior={activeBayesianData.sigma_prior}
                  priorLabel={activeBayesianData.prior_name}
                  muAudio={activeBayesianData.mu_audio}
                  sigmaAudio={activeBayesianData.sigma_audio}
                  muVisual={activeBayesianData.mu_visual}
                  sigmaVisual={activeBayesianData.sigma_visual}
                  muSensory={activeBayesianData.muSensory}
                  sigmaSensory={activeBayesianData.sigmaSensory}
                  muPost={activeBayesianData.muPost}
                  sigmaPost={activeBayesianData.sigmaPost}
                  height={195}
                  showPrior={showPrior}
                  showAudio={showA}
                  showVisual={showV}
                  showSensory={showAV}
                  showPost={showPost}
                />
              </div>

              {/* 2. 2D AFFECTIVE CIRCUMPLEX (Replaces Sliders/Bars Below) */}
              <div className="glass-panel" style={{ padding: '0.9rem 1.1rem' }}>
                <AffectiveCircumplex
                  valence={valenceData}
                  arousal={arousalData}
                  height={300}
                  priorLabel={valenceData.prior_name?.split(' ')[0] || 'Prior'}
                  showPosterior={circShowPost}
                  setShowPosterior={setCircShowPost}
                  showSensory={circShowSensory}
                  setShowSensory={setCircShowSensory}
                  showPrior={circShowPrior}
                  setShowPrior={setCircShowPrior}
                />
              </div>
            </>
          )}
        </div>

        {/* ── RIGHT COLUMN (Robot / Output & Control) ─────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* 1. iCub Robot Simulation (Overlay Placeholder Area) */}
          <div
            className="glass-panel"
            style={{
              padding: '1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Bot size={15} color="var(--accent-primary)" />
                iCub Robot Simulation
              </span>
              <span style={{ fontSize: '0.68rem', color: '#9ca3af', fontWeight: 600 }}>
                PyBullet Window Overlay
              </span>
            </div>

            {/* Designated Overlay Placeholder Viewport */}
            <div
              style={{
                width: '100%',
                height: '320px',
                borderRadius: '0.6rem',
                border: '1.5px dashed rgba(99, 102, 241, 0.4)',
                background: 'rgba(5, 8, 16, 0.75)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.6rem',
                position: 'relative',
              }}
            >
              {/* Corner Alignment Crosses */}
              <div style={{ position: 'absolute', top: 8, left: 8, color: 'rgba(99, 102, 241, 0.5)', fontSize: '10px' }}>┌</div>
              <div style={{ position: 'absolute', top: 8, right: 8, color: 'rgba(99, 102, 241, 0.5)', fontSize: '10px' }}>┐</div>
              <div style={{ position: 'absolute', bottom: 8, left: 8, color: 'rgba(99, 102, 241, 0.5)', fontSize: '10px' }}>└</div>
              <div style={{ position: 'absolute', bottom: 8, right: 8, color: 'rgba(99, 102, 241, 0.5)', fontSize: '10px' }}>┘</div>

              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: '50%',
                  background: 'rgba(99, 102, 241, 0.12)',
                  border: '1px solid rgba(99, 102, 241, 0.3)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Bot size={24} color="var(--accent-primary)" />
              </div>

              <div style={{ textAlign: 'center', padding: '0 1rem' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#e0e7ff', marginBottom: '0.2rem' }}>
                  iCub 3D Simulation Overlay Space
                </div>
                <div style={{ fontSize: '0.68rem', color: '#6b7280', lineHeight: 1.35 }}>
                  Align the PyBullet upper-body physical simulation window over this container.
                </div>
              </div>
            </div>
          </div>

          {/* 2. Robot Speech & Policy Output */}
          <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Sparkles size={14} color="var(--accent-primary)" />
                Robot Speech & Policy
              </span>
              {latestAssistantTurn?.gesture && (
                <span
                  style={{
                    fontSize: '0.68rem',
                    padding: '0.15rem 0.5rem',
                    borderRadius: '9999px',
                    background: 'rgba(99, 102, 241, 0.15)',
                    color: '#a5b4fc',
                    border: '1px solid rgba(99, 102, 241, 0.3)',
                    fontWeight: 700,
                  }}
                >
                  Gesture: {latestAssistantTurn.gesture.toUpperCase()}
                </span>
              )}
            </div>

            {/* Policy Badge (Label Only) */}
            {latestAssistantTurn?.policy ? (
              <div
                style={{
                  padding: '0.4rem 0.65rem',
                  borderRadius: '0.45rem',
                  background:
                    latestAssistantTurn.policy.action === 'clarify'
                      ? 'rgba(244, 63, 94, 0.12)'
                      : latestAssistantTurn.policy.action === 'cautious'
                      ? 'rgba(245, 158, 11, 0.12)'
                      : 'rgba(16, 185, 129, 0.12)',
                  border: `1px solid ${
                    latestAssistantTurn.policy.action === 'clarify'
                      ? 'rgba(244, 63, 94, 0.35)'
                      : latestAssistantTurn.policy.action === 'cautious'
                      ? 'rgba(245, 158, 11, 0.35)'
                      : 'rgba(16, 185, 129, 0.35)'
                  }`,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', fontWeight: 800 }}>
                  {latestAssistantTurn.policy.action === 'clarify' ? (
                    <HelpCircle size={14} color="#fb7185" />
                  ) : latestAssistantTurn.policy.action === 'cautious' ? (
                    <Eye size={14} color="#fbbf24" />
                  ) : (
                    <CheckCircle2 size={14} color="#34d399" />
                  )}
                  <span
                    style={{
                      color:
                        latestAssistantTurn.policy.action === 'clarify'
                          ? '#fb7185'
                          : latestAssistantTurn.policy.action === 'cautious'
                          ? '#fbbf24'
                          : '#34d399',
                    }}
                  >
                    Policy: {latestAssistantTurn.policy.label}
                  </span>
                </div>
              </div>
            ) : (
              <div style={{ fontSize: '0.72rem', color: '#6b7280', padding: '0.2rem 0' }}>
                Awaiting first turn policy decision...
              </div>
            )}

            {/* Robot Spoken Output */}
            <div
              style={{
                padding: '0.75rem 0.85rem',
                borderRadius: '0.5rem',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid var(--border-subtle)',
                minHeight: '65px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
              }}
            >
              {latestAssistantTurn ? (
                latestAssistantTurn.text.startsWith('[No text generated') ? (
                  <div style={{ color: '#f87171', fontSize: '0.82rem', lineHeight: 1.45, display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    <div style={{ fontWeight: 800, display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#ef4444' }}>
                      <AlertCircle size={15} />
                      LLM Response Notice
                    </div>
                    <div style={{ color: '#fca5a5', fontSize: '0.78rem' }}>
                      {latestAssistantTurn.text}
                    </div>
                    <button
                      type="button"
                      onClick={() => setIsSettingsOpen(true)}
                      style={{
                        alignSelf: 'flex-start',
                        padding: '0.2rem 0.6rem',
                        fontSize: '0.72rem',
                        borderRadius: '0.35rem',
                        background: 'rgba(239, 68, 68, 0.2)',
                        border: '1px solid rgba(239, 68, 68, 0.4)',
                        color: '#fecaca',
                        cursor: 'pointer',
                        fontWeight: 700,
                      }}
                    >
                      Open Settings to configure model / endpoint
                    </button>
                  </div>
                ) : (
                  <div style={{ color: '#ffffff', fontSize: '0.88rem', lineHeight: 1.45 }}>
                    &ldquo;{latestAssistantTurn.text}&rdquo;
                  </div>
                )
              ) : (
                <div style={{ color: '#6b7280', fontSize: '0.8rem', textAlign: 'center' }}>
                  Robot response will appear here after speaking.
                </div>
              )}
            </div>

            {/* Audio Feedback Player */}
            {latestAssistantTurn?.audioUrl && (
              <div style={{ marginTop: '0.2rem' }}>
                <audio
                  ref={audioPlayerRef}
                  controls
                  autoPlay={shouldAutoplayAudio}
                  src={latestAssistantTurn.audioUrl}
                  onPlay={() => {
                    vadControllerRef.current?.setRobotSpeaking(true);
                  }}
                  onEnded={() => {
                    setTimeout(() => {
                      vadControllerRef.current?.setRobotSpeaking(false);
                    }, 400);
                  }}
                  onError={() => {
                    vadControllerRef.current?.setRobotSpeaking(false);
                  }}
                  style={{ width: '100%', height: '28px' }}
                />
              </div>
            )}
          </div>

          {/* 3. 🎤 SPEAK TO ROBOT (Record Action Controller) */}
          <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {/* Header & Recording Mode Segmented Switcher */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Mic size={15} color="var(--accent-primary)" />
                Audio & Video Capture
              </span>

              {/* Mode Toggle Pills */}
              <div
                style={{
                  display: 'flex',
                  background: 'rgba(255, 255, 255, 0.06)',
                  borderRadius: '9999px',
                  padding: '0.15rem',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <button
                  type="button"
                  onClick={() => setRecordingMode('manual')}
                  style={{
                    padding: '0.2rem 0.6rem',
                    borderRadius: '9999px',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    border: 'none',
                    cursor: 'pointer',
                    background: recordingMode === 'manual' ? 'var(--accent-primary)' : 'transparent',
                    color: recordingMode === 'manual' ? '#ffffff' : '#9ca3af',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <Mic size={11} />
                  Click to Record
                </button>
                <button
                  type="button"
                  onClick={() => setRecordingMode('vad')}
                  style={{
                    padding: '0.2rem 0.6rem',
                    borderRadius: '9999px',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    border: 'none',
                    cursor: 'pointer',
                    background: recordingMode === 'vad' ? '#10b981' : 'transparent',
                    color: recordingMode === 'vad' ? '#ffffff' : '#9ca3af',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <AudioWaveform size={11} />
                  Continuous VAD
                </button>
              </div>
            </div>

            {/* MANUAL MODE VIEW */}
            {recordingMode === 'manual' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {!isRecording ? (
                  <button
                    type="button"
                    onClick={startRecording}
                    disabled={processing}
                    style={{
                      width: '100%',
                      padding: '0.85rem 1rem',
                      borderRadius: '0.6rem',
                      background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
                      color: '#ffffff',
                      fontWeight: 800,
                      fontSize: '0.92rem',
                      border: 'none',
                      cursor: processing ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '0.5rem',
                      boxShadow: '0 4px 16px rgba(99, 102, 241, 0.4)',
                      transition: 'all 0.15s ease',
                      opacity: processing ? 0.7 : 1,
                    }}
                  >
                    <Mic size={18} />
                    Speak to Robot (Click to Record)
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stopRecording}
                    style={{
                      width: '100%',
                      padding: '0.85rem 1rem',
                      borderRadius: '0.6rem',
                      background: '#ef4444',
                      color: '#ffffff',
                      fontWeight: 800,
                      fontSize: '0.92rem',
                      border: 'none',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '0.5rem',
                      boxShadow: '0 4px 16px rgba(239, 68, 68, 0.5)',
                      animation: 'pulse 1.5s infinite',
                    }}
                  >
                    <Radio size={18} />
                    Stop & Process Turn
                  </button>
                )}
                <div style={{ fontSize: '0.7rem', color: '#9ca3af', textAlign: 'center' }}>
                  Click to start multimodal capture; click again when finished speaking.
                </div>
              </div>
            )}

            {/* CONTINUOUS VAD MODE VIEW */}
            {recordingMode === 'vad' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {/* VAD Dynamic Status Banner */}
                <div
                  style={{
                    padding: '0.55rem 0.75rem',
                    borderRadius: '0.5rem',
                    background:
                      vadState === 'SPEECH_DETECTED'
                        ? 'rgba(239, 68, 68, 0.18)'
                        : vadState === 'PROCESSING'
                        ? 'rgba(245, 158, 11, 0.15)'
                        : vadState === 'ROBOT_SPEAKING'
                        ? 'rgba(168, 85, 247, 0.15)'
                        : 'rgba(16, 185, 129, 0.12)',
                    border: `1px solid ${
                      vadState === 'SPEECH_DETECTED'
                        ? 'rgba(239, 68, 68, 0.5)'
                        : vadState === 'PROCESSING'
                        ? 'rgba(245, 158, 11, 0.4)'
                        : vadState === 'ROBOT_SPEAKING'
                        ? 'rgba(168, 85, 247, 0.4)'
                        : 'rgba(16, 185, 129, 0.35)'
                    }`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background:
                          vadState === 'SPEECH_DETECTED'
                            ? '#ef4444'
                            : vadState === 'PROCESSING'
                            ? '#f59e0b'
                            : vadState === 'ROBOT_SPEAKING'
                            ? '#c084fc'
                            : '#10b981',
                        animation: vadState === 'SPEECH_DETECTED' || vadState === 'LISTENING' ? 'pulse 1.2s infinite' : 'none',
                      }}
                    />
                    <span
                      style={{
                        fontSize: '0.75rem',
                        fontWeight: 800,
                        color:
                          vadState === 'SPEECH_DETECTED'
                            ? '#fca5a5'
                            : vadState === 'PROCESSING'
                            ? '#fde68a'
                            : vadState === 'ROBOT_SPEAKING'
                            ? '#e9d5ff'
                            : '#6ee7b7',
                      }}
                    >
                      {vadState === 'SPEECH_DETECTED'
                        ? 'Speech Detected (Recording...)'
                        : vadState === 'PROCESSING'
                        ? 'Processing Multimodal Turn...'
                        : vadState === 'ROBOT_SPEAKING'
                        ? 'Robot Speaking (Muted)'
                        : 'VAD Hands-Free: Listening...'}
                    </span>
                  </div>
                  <span style={{ fontSize: '0.66rem', color: '#9ca3af' }}>
                    {silenceHangoverMs}ms Hangover
                  </span>
                </div>

                {/* Real-Time Live Energy Meter */}
                <div
                  style={{
                    background: 'rgba(0, 0, 0, 0.35)',
                    padding: '0.45rem 0.65rem',
                    borderRadius: '0.4rem',
                    border: '1px solid var(--border-subtle)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: '#9ca3af', marginBottom: '0.25rem' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                      <Volume2 size={11} color="var(--accent-primary)" />
                      Voice Input Level:
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>
                      {(audioRms * 1000).toFixed(0)} / {(audioThreshold * 1000).toFixed(0)} th
                    </span>
                  </div>
                  {/* Energy Meter Bar */}
                  <div
                    style={{
                      position: 'relative',
                      width: '100%',
                      height: '7px',
                      borderRadius: '4px',
                      background: 'rgba(255, 255, 255, 0.08)',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Active RMS fill */}
                    <div
                      style={{
                        height: '100%',
                        width: `${Math.min(100, (audioRms / 0.05) * 100)}%`,
                        background: audioRms > audioThreshold ? '#ef4444' : '#10b981',
                        borderRadius: '4px',
                        transition: 'width 0.05s ease-out',
                      }}
                    />
                    {/* Threshold marker tick */}
                    <div
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        left: `${Math.min(100, (audioThreshold / 0.05) * 100)}%`,
                        width: '2px',
                        background: '#ffffff',
                        boxShadow: '0 0 4px #ffffff',
                      }}
                    />
                  </div>
                </div>

                {/* VAD Action Buttons */}
                {isRecording ? (
                  <div style={{ display: 'flex', gap: '0.45rem' }}>
                    <button
                      type="button"
                      onClick={stopRecording}
                      style={{
                        flex: 1,
                        padding: '0.65rem 0.75rem',
                        borderRadius: '0.5rem',
                        background: '#ef4444',
                        color: '#ffffff',
                        fontWeight: 800,
                        fontSize: '0.8rem',
                        border: 'none',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '0.35rem',
                      }}
                    >
                      <Radio size={14} />
                      Stop & Commit Now
                    </button>
                    <button
                      type="button"
                      onClick={cancelRecording}
                      style={{
                        padding: '0.65rem 0.85rem',
                        borderRadius: '0.5rem',
                        background: 'rgba(255, 255, 255, 0.08)',
                        border: '1px solid var(--border-subtle)',
                        color: '#d1d5db',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={startRecording}
                    disabled={processing || vadState === 'ROBOT_SPEAKING'}
                    style={{
                      width: '100%',
                      padding: '0.65rem 0.85rem',
                      borderRadius: '0.5rem',
                      background: 'rgba(255, 255, 255, 0.07)',
                      border: '1px solid var(--border-subtle)',
                      color: '#d1d5db',
                      fontWeight: 700,
                      fontSize: '0.8rem',
                      cursor: processing ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '0.4rem',
                      transition: 'all 0.15s ease',
                      opacity: processing ? 0.6 : 1,
                    }}
                  >
                    <Zap size={13} color="var(--accent-primary)" />
                    Push to Talk (Manual Override)
                  </button>
                )}
              </div>
            )}

            {/* Processing Spinner / Step Indicator */}
            {processing && (
              <div
                style={{
                  padding: '0.5rem 0.75rem',
                  borderRadius: '0.4rem',
                  background: 'rgba(99, 102, 241, 0.15)',
                  border: '1px solid var(--accent-primary)',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '0.75rem', color: '#c7d2fe', fontWeight: 600 }}>
                  {processingStep}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── INTERACTION FLOW LOGGING PANEL ────────────────────────────── */}
      <InteractionLogTable
        logs={loggedTurns}
        enableLogging={enableLogging}
        onToggleLogging={(enabled) => setEnableLogging(enabled)}
        onClearLogs={handleClearLogs}
        onDeleteLastTurn={handleDeleteLastTurn}
      />

      {/* ── MODAL POP-UP: Precision Sliders & Bayesian Source Weights ─────── */}
      {isPrecisionModalOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 100,
            background: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(10px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
          }}
          onClick={() => setIsPrecisionModalOpen(false)}
        >
          <div
            className="glass-panel"
            style={{
              maxWidth: '680px',
              width: '100%',
              padding: '1.5rem 1.75rem',
              background: 'rgba(17, 24, 39, 0.95)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              boxShadow: '0 20px 50px rgba(0, 0, 0, 0.7)',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.25rem',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sliders size={18} color="var(--accent-primary)" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#ffffff' }}>
                  Interactive Precision Controls & Source Weights
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setIsPrecisionModalOpen(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  border: 'none',
                  borderRadius: '0.35rem',
                  color: '#9ca3af',
                  padding: '0.35rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <X size={18} />
              </button>
            </div>

            {/* Sliders Grid */}
            {activeBayesianData ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {/* 1. Prior Precision Slider (Red) */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                    <span style={{ color: '#fca5a5', fontWeight: 700 }}>
                      Prior Precision τ_prior ({activeBayesianData.prior_name?.split(' ')[0]})
                    </span>
                    <span style={{ color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                      {activeBayesianData.tauPrior.toFixed(1)} (σ = {activeBayesianData.sigma_prior.toFixed(2)})
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="40.0"
                    step="0.5"
                    value={activeBayesianData.tauPrior}
                    onChange={(e) => handleTauPriorChange(parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#ef4444' }}
                  />
                </div>

                {/* 2. Audio Precision Slider (Blue) */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                    <span style={{ color: '#93c5fd', fontWeight: 700 }}>
                      Acoustic Precision τ_audio (Wav2Vec2)
                    </span>
                    <span style={{ color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                      {activeBayesianData.tauAudio.toFixed(1)} (σ = {activeBayesianData.sigma_audio.toFixed(2)})
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="40.0"
                    step="0.5"
                    value={activeBayesianData.tauAudio}
                    onChange={(e) => handleTauAudioChange(parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#3b82f6' }}
                  />
                </div>

                {/* 3. Visual Precision Slider (Amber) */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                    <span style={{ color: '#fde68a', fontWeight: 700 }}>
                      Visual Precision τ_visual (EmoNet)
                    </span>
                    <span style={{ color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                      {activeBayesianData.tauVisual.toFixed(1)} (σ = {activeBayesianData.sigma_visual.toFixed(2)})
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="40.0"
                    step="0.5"
                    value={activeBayesianData.tauVisual}
                    onChange={(e) => handleTauVisualChange(parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#f59e0b' }}
                  />
                </div>

                {/* 4. Combined Sensory Precision Slider (Green AV) */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                    <span style={{ color: '#6ee7b7', fontWeight: 700 }}>
                      Combined Sensory Precision τ_sensory (A + V)
                    </span>
                    <span style={{ color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                      {activeBayesianData.tauSensory.toFixed(1)} (σ = {activeBayesianData.sigmaSensory.toFixed(2)})
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="80.0"
                    step="1.0"
                    value={activeBayesianData.tauSensory}
                    onChange={(e) => handleTauSensoryChange(parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#10b981' }}
                  />
                </div>

                {/* Bayesian Source Weight Breakdown Bars */}
                <div style={{ marginTop: '0.5rem', padding: '1rem', borderRadius: '0.5rem', background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, marginBottom: '0.6rem', color: '#e5e7eb' }}>
                    Relative PWP Posterior Source Weights:
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.75rem' }}>
                    {/* Prior Bar (Red) */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                        <span style={{ color: '#fca5a5' }}>Prior Weight (w_prior):</span>
                        <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{(activeBayesianData.wPrior * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${activeBayesianData.wPrior * 100}%`, height: '100%', background: '#ef4444' }} />
                      </div>
                    </div>

                    {/* Audio Bar (Blue) */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                        <span style={{ color: '#93c5fd' }}>Audio Likelihood Weight (w_audio):</span>
                        <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{(activeBayesianData.wAudio * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${activeBayesianData.wAudio * 100}%`, height: '100%', background: '#3b82f6' }} />
                      </div>
                    </div>

                    {/* Visual Bar (Amber) */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                        <span style={{ color: '#fde68a' }}>Visual Likelihood Weight (w_visual):</span>
                        <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{(activeBayesianData.wVisual * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${activeBayesianData.wVisual * 100}%`, height: '100%', background: '#f59e0b' }} />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Reset Sliders Button */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.25rem' }}>
                  <button
                    type="button"
                    onClick={resetPrecisionSliders}
                    style={{
                      padding: '0.4rem 0.9rem',
                      borderRadius: '0.4rem',
                      background: 'rgba(255, 255, 255, 0.08)',
                      border: '1px solid var(--border-subtle)',
                      color: '#d1d5db',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Reset Model Precisions to Defaults
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '1.5rem', color: '#9ca3af', fontSize: '0.85rem' }}>
                No active turn recorded. Precision overrides will apply once conversational turns begin.
              </div>
            )}

            {/* VAD Continuous Recording Configuration Section (Always accessible) */}
            <div
              style={{
                marginTop: '0.5rem',
                padding: '1rem',
                borderRadius: '0.6rem',
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.85rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#6ee7b7', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <AudioWaveform size={15} color="#10b981" />
                  Voice Activity Detection (VAD) Tuning
                </span>
                <span style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                  Mode: {recordingMode === 'vad' ? 'Continuous Hands-Free' : 'Manual Push-to-Talk'}
                </span>
              </div>

              {/* VAD Sensitivity Slider */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.25rem' }}>
                  <span style={{ color: '#d1d5db', fontWeight: 600 }}>VAD Trigger Sensitivity</span>
                  <span style={{ color: '#6ee7b7', fontWeight: 700 }}>
                    Level {vadSensitivity} of 5 &mdash; {
                      vadSensitivity === 1
                        ? 'Low (Noisy Room)'
                        : vadSensitivity === 2
                        ? 'Medium-Low'
                        : vadSensitivity === 3
                        ? 'Medium (Balanced Default)'
                        : vadSensitivity === 4
                        ? 'Medium-High'
                        : 'High (Quiet Room)'
                    }
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="1"
                  value={vadSensitivity}
                  onChange={(e) => setVadSensitivity(parseInt(e.target.value, 10))}
                  style={{ width: '100%', accentColor: '#10b981' }}
                />
              </div>

              {/* Silence Hangover Slider */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.25rem' }}>
                  <span style={{ color: '#d1d5db', fontWeight: 600 }}>Silence Hangover Duration</span>
                  <span style={{ color: '#6ee7b7', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                    {silenceHangoverMs} ms ({(silenceHangoverMs / 1000).toFixed(2)}s)
                  </span>
                </div>
                <input
                  type="range"
                  min="400"
                  max="2000"
                  step="50"
                  value={silenceHangoverMs}
                  onChange={(e) => setSilenceHangoverMs(parseInt(e.target.value, 10))}
                  style={{ width: '100%', accentColor: '#10b981' }}
                />
                <span style={{ fontSize: '0.68rem', color: '#9ca3af', marginTop: '0.15rem', display: 'block' }}>
                  Silence duration required after you stop speaking before the turn automatically finalizes and sends.
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* ── SETTINGS SIDEBAR DRAWER ────────────────────────────────────────── */}
      {isSettingsOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 120,
            background: 'rgba(0, 0, 0, 0.65)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            justifyContent: 'flex-end',
          }}
          onClick={() => setIsSettingsOpen(false)}
        >
          <div
            style={{
              width: '460px',
              maxWidth: '92vw',
              height: '100%',
              background: '#0d131f',
              borderLeft: '1px solid rgba(255, 255, 255, 0.12)',
              boxShadow: '-10px 0 35px rgba(0, 0, 0, 0.8)',
              display: 'flex',
              flexDirection: 'column',
              padding: '1.5rem',
              overflowY: 'auto',
              gap: '1.25rem',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Drawer Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.85rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Settings size={20} color="var(--accent-primary)" />
                <div>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: '#ffffff', margin: 0 }}>
                    Studio Settings
                  </h3>
                  <span style={{ fontSize: '0.72rem', color: '#9ca3af' }}>
                    Configure LLM model, endpoint, fallbacks, and audio.
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  border: 'none',
                  borderRadius: '0.35rem',
                  color: '#9ca3af',
                  padding: '0.35rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <X size={18} />
              </button>
            </div>

            {/* SECTION 1: LLM CONFIGURATION */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#a5b4fc', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                1. Language Model (LLM) Engine
              </span>

              {/* Endpoint Preset Buttons */}
              <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={() => {
                    const ep = 'http://127.0.0.1:11434';
                    setLlmApiBase(ep);
                    localStorage.setItem('icub_llm_api_base', ep);
                    checkLlmConnection(ep);
                  }}
                  style={{
                    padding: '0.25rem 0.55rem',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    borderRadius: '0.3rem',
                    background: llmApiBase.includes('11434') ? 'var(--accent-primary)' : 'rgba(255, 255, 255, 0.06)',
                    color: llmApiBase.includes('11434') ? '#ffffff' : '#9ca3af',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                  }}
                >
                  Ollama (127.0.0.1:11434)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const ep = 'http://127.0.0.1:8080/v1';
                    setLlmApiBase(ep);
                    localStorage.setItem('icub_llm_api_base', ep);
                    checkLlmConnection(ep);
                  }}
                  style={{
                    padding: '0.25rem 0.55rem',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    borderRadius: '0.3rem',
                    background: llmApiBase.includes('8080') ? 'var(--accent-primary)' : 'rgba(255, 255, 255, 0.06)',
                    color: llmApiBase.includes('8080') ? '#ffffff' : '#9ca3af',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer',
                  }}
                >
                  llama.cpp (127.0.0.1:8080)
                </button>
              </div>

              {/* Endpoint URL Input */}
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#d1d5db', marginBottom: '0.25rem' }}>
                  API Base Endpoint
                </label>
                <input
                  type="text"
                  value={llmApiBase}
                  onChange={(e) => {
                    setLlmApiBase(e.target.value);
                    localStorage.setItem('icub_llm_api_base', e.target.value);
                  }}
                  placeholder="e.g. http://127.0.0.1:11434"
                  style={{
                    width: '100%',
                    padding: '0.45rem 0.65rem',
                    borderRadius: '0.35rem',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-subtle)',
                    color: '#ffffff',
                    fontSize: '0.78rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>

              {/* Model Selection */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#d1d5db' }}>
                    Active Model Name
                  </label>
                  <button
                    type="button"
                    onClick={() => checkLlmConnection()}
                    disabled={llmConnStatus.checking}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: 'var(--accent-primary)',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.25rem',
                    }}
                  >
                    <RefreshCw size={11} className={llmConnStatus.checking ? 'animate-spin' : ''} />
                    Refresh Models
                  </button>
                </div>

                {availableModels.length > 0 ? (
                  <select
                    value={selectedModel}
                    onChange={(e) => {
                      setSelectedModel(e.target.value);
                      localStorage.setItem('icub_selected_model', e.target.value);
                    }}
                    style={{
                      width: '100%',
                      padding: '0.45rem 0.65rem',
                      borderRadius: '0.35rem',
                      background: '#111827',
                      border: '1px solid var(--border-subtle)',
                      color: '#ffffff',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      marginBottom: '0.35rem',
                    }}
                  >
                    {availableModels.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                ) : null}

                {/* Custom Model Name Input */}
                <input
                  type="text"
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    localStorage.setItem('icub_selected_model', e.target.value);
                  }}
                  placeholder="e.g. qwen3:0.6b, qwen2.5:3b"
                  style={{
                    width: '100%',
                    padding: '0.45rem 0.65rem',
                    borderRadius: '0.35rem',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-subtle)',
                    color: '#ffffff',
                    fontSize: '0.78rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                />
              </div>

              {/* Optional API Key */}
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#d1d5db', marginBottom: '0.25rem' }}>
                  API Key (Optional for local Ollama)
                </label>
                <input
                  type="password"
                  value={llmApiKey}
                  onChange={(e) => {
                    setLlmApiKey(e.target.value);
                    localStorage.setItem('icub_llm_api_key', e.target.value);
                  }}
                  placeholder="sk-..."
                  style={{
                    width: '100%',
                    padding: '0.45rem 0.65rem',
                    borderRadius: '0.35rem',
                    background: 'rgba(0,0,0,0.4)',
                    border: '1px solid var(--border-subtle)',
                    color: '#ffffff',
                    fontSize: '0.78rem',
                  }}
                />
              </div>

              {/* Test Connection Button & Status */}
              <div>
                <button
                  type="button"
                  onClick={() => checkLlmConnection()}
                  disabled={llmConnStatus.checking}
                  style={{
                    width: '100%',
                    padding: '0.5rem',
                    borderRadius: '0.35rem',
                    background: 'rgba(99, 102, 241, 0.2)',
                    border: '1px solid var(--accent-primary)',
                    color: '#ffffff',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    cursor: llmConnStatus.checking ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.4rem',
                  }}
                >
                  <Server size={14} />
                  {llmConnStatus.checking ? 'Pinging Endpoint...' : 'Test Connection'}
                </button>

                {llmConnStatus.message && (
                  <div
                    style={{
                      marginTop: '0.4rem',
                      padding: '0.4rem 0.65rem',
                      borderRadius: '0.35rem',
                      fontSize: '0.72rem',
                      background: llmConnStatus.success ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                      border: `1px solid ${llmConnStatus.success ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'}`,
                      color: llmConnStatus.success ? '#6ee7b7' : '#fca5a5',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.35rem',
                    }}
                  >
                    {llmConnStatus.success ? <Check size={13} /> : <AlertCircle size={13} />}
                    {llmConnStatus.message}
                  </div>
                )}
              </div>
            </div>

            {/* SECTION 2: FALLBACK TEXT BEHAVIOR */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                2. Fallback Response Mode
              </span>
              <span style={{ fontSize: '0.72rem', color: '#9ca3af' }}>
                What happens when the LLM is unreachable or produces no text:
              </span>

              <label style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', cursor: 'pointer', fontSize: '0.78rem', color: '#d1d5db' }}>
                <input
                  type="radio"
                  name="fallback_mode"
                  checked={fallbackMode === 'error'}
                  onChange={() => {
                    setFallbackMode('error');
                    localStorage.setItem('icub_fallback_mode', 'error');
                  }}
                  style={{ marginTop: '0.15rem', accentColor: 'var(--accent-primary)' }}
                />
                <div>
                  <strong style={{ color: '#ffffff' }}>Explicit Warning [No text generated] (Recommended)</strong>
                  <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                    Clearly indicates LLM failure and suppresses empty TTS audio playback.
                  </div>
                </div>
              </label>

              <label style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', cursor: 'pointer', fontSize: '0.78rem', color: '#d1d5db' }}>
                <input
                  type="radio"
                  name="fallback_mode"
                  checked={fallbackMode === 'canned'}
                  onChange={() => {
                    setFallbackMode('canned');
                    localStorage.setItem('icub_fallback_mode', 'canned');
                  }}
                  style={{ marginTop: '0.15rem', accentColor: 'var(--accent-primary)' }}
                />
                <div>
                  <strong style={{ color: '#ffffff' }}>Friendly Static Phrase</strong>
                  <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                    Speaks &ldquo;Thank you for sharing that with me. I understand.&rdquo;
                  </div>
                </div>
              </label>
            </div>

            {/* SECTION 3: SPEECH & AUDIO */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                3. Speech (STT & TTS)
              </span>

              {/* Whisper Model Size */}
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#d1d5db', marginBottom: '0.25rem' }}>
                  Faster-Whisper Model Size
                </label>
                <select
                  value={whisperSize}
                  onChange={(e) => {
                    setWhisperSize(e.target.value);
                    localStorage.setItem('icub_whisper_size', e.target.value);
                  }}
                  style={{
                    width: '100%',
                    padding: '0.45rem 0.65rem',
                    borderRadius: '0.35rem',
                    background: '#111827',
                    border: '1px solid var(--border-subtle)',
                    color: '#ffffff',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                  }}
                >
                  <option value="tiny">Tiny (Fastest, Lowest VRAM)</option>
                  <option value="base">Base (Fast)</option>
                  <option value="small">Small (Balanced Default)</option>
                  <option value="medium">Medium (Highest Accuracy)</option>
                </select>
              </div>

              {/* Kokoro TTS Voice */}
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 600, color: '#d1d5db', marginBottom: '0.25rem' }}>
                  Kokoro-82M Voice
                </label>
                <select
                  value={ttsVoice}
                  onChange={(e) => {
                    setTtsVoice(e.target.value);
                    localStorage.setItem('icub_tts_voice', e.target.value);
                  }}
                  style={{
                    width: '100%',
                    padding: '0.45rem 0.65rem',
                    borderRadius: '0.35rem',
                    background: '#111827',
                    border: '1px solid var(--border-subtle)',
                    color: '#ffffff',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                  }}
                >
                  <option value="af_heart">af_heart (US Female Warm)</option>
                  <option value="am_adam">am_adam (US Male)</option>
                  <option value="bf_alice">bf_alice (UK Female)</option>
                  <option value="bm_george">bm_george (UK Male)</option>
                  <option value="jf_alpha">jf_alpha (Japanese)</option>
                </select>
              </div>
            </div>

            {/* SECTION 4: CLOSE BUTTON */}
            <div style={{ marginTop: 'auto', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem', display: 'flex', gap: '0.5rem' }}>
              <button
                type="button"
                onClick={() => {
                  localStorage.removeItem('icub_llm_api_base');
                  localStorage.removeItem('icub_selected_model');
                  localStorage.removeItem('icub_fallback_mode');
                  localStorage.removeItem('icub_whisper_size');
                  setLlmApiBase('http://127.0.0.1:11434');
                  setSelectedModel('qwen3:0.6b');
                  setFallbackMode('error');
                  setWhisperSize('small');
                  checkLlmConnection('http://127.0.0.1:11434');
                }}
                style={{
                  padding: '0.55rem 0.85rem',
                  borderRadius: '0.4rem',
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid var(--border-subtle)',
                  color: '#9ca3af',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Reset Defaults
              </button>
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                style={{
                  flex: 1,
                  padding: '0.55rem 1rem',
                  borderRadius: '0.4rem',
                  background: 'var(--accent-primary)',
                  border: 'none',
                  color: '#ffffff',
                  fontSize: '0.82rem',
                  fontWeight: 800,
                  cursor: 'pointer',
                }}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
